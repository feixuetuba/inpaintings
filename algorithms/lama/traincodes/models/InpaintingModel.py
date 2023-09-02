import logging
import os
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torch import functional as F
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from algorithms import BaseModel
from algorithms.lama.networks import load_network
from algorithms.lama.traincodes.dataset import BuildDataloader
from algorithms.lama.traincodes.losses.adversarial import make_discrim_loss
from algorithms.lama.traincodes.losses.feature_matching import masked_l1_loss, feature_matching_loss
from algorithms.lama.traincodes.losses.perceptual import PerceptualLoss, ResNetPL
from algorithms.lama.traincodes.utils import get_ramp, make_mask_distance_weighter, make_optimizer
from utils import ddp
from utils.configure import Configure
from utils.ddp import is_main_process


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)



class InpaintingModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._prepared = False
        self.device = kwargs.get("device", torch.device("cuda"))
        self.losses = {}

    @classmethod
    def get_argparser(cls):
        parser = ArgumentParser();
        parser.add_argument("cfg", type=str, help="config file")
        parser.add_argument("--which_epoch", type=int, default=0, help="epoch begin")
        parser.add_argument("-g", "--generator_pretrained", type=str, default=None, help="Generator pretrained parameters")
        parser.add_argument("-d", "--discriminator_pretrained", type=str, default=None, help="Discriminator pretrained parameters")
        parser.add_argument("-s", "--save_dir", type=str, default="experiments", help="Dircetory to save results")
        parser.add_argument("--total_epoch", type=int, default=200, help="total epoch")
        return parser

    def prepare(self, opts, save_dir, cudas=[0,1,2,3], use_ddp=True, image_to_discriminator='predicted_image',
                rescale_scheduler_kwargs=None,
                add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                *args, **kwargs):
        cfg = Configure(opts.cfg, TORCH_HOME=os.environ["TORCH_HOME"])
        self.cfg = cfg
        self.config = cfg.model
        self.cfg.use_ddp = use_ddp
        self.cfg.cudas = cudas
        txt_logs_dir = os.path.join(save_dir, "txts")
        ckpt_save_dir = os.path.join(save_dir, "ckeckpoints")
        sample_save_dir = os.path.join(save_dir, "samples")
        log_save_dir = os.path.join(save_dir, "logs")

        os.makedirs(txt_logs_dir, exist_ok=True)
        os.makedirs(ckpt_save_dir, exist_ok=True)
        os.makedirs(sample_save_dir, exist_ok=True)
        os.makedirs(log_save_dir, exist_ok=True)

        self.config["txt_logs_dir"] = txt_logs_dir
        self.config["ckpt_save_dir"] = ckpt_save_dir
        self.config["sample_save_dir"] = sample_save_dir
        self.config["log_save_dir"] = log_save_dir

        # my_dir = os.path.dirname(os.path.abspath(__file__))
        # cfg_file = kwargs.get("cfg", my_dir + "/big-lama.yaml")
        # cfg = Configure(cfg=cfg_file)
        self.generator = load_network("generator", cfg)
        if opts.generator_pretrained is not None:
            if ddp.is_main_process():
                logging.info(f"Generator load pretrain from: {opts.generator_pretrained}")
            state_dict = torch.load(opts.generator_pretrained)
            self.generator.load_state_dict(state_dict)
        self.generator.to(self.device)

        if "perceptual" in self.config.losses and self.config.losses.perceptual.weight > 0:
            self.losses["perceptual"] = PerceptualLoss()

        if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
            self.loss_l1 = nn.L1Loss(reduction='none')

        if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
            self.loss_mse = nn.MSELoss(reduction='none')

        if "resnet_pl" in self.config.losses and self.config.losses.resnet_pl.weight > 0:
            self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl).to(self.device)

        if "adversarial" in self.config.losses:
            self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)

        self.discriminator = load_network("discriminator", cfg)
        if opts.discriminator_pretrained is not None:
            if ddp.is_main_process():
                logging.info(f"Discriminator load pretrain from: {opts.discriminator_pretrained}")
            state_dict = torch.load(opts.discriminator_pretrained)
            self.discriminator.load_state_dict(state_dict)
        self.discriminator.to(self.device)
        self.train_set = BuildDataloader(cfg)
        self.val_set = None
        if "val" in cfg:
            self.val_set = BuildDataloader(cfg, "val")

        self.start_epoch = kwargs.get("epoch", 0)
        # self.generator_optim =

        self.concat_mask = self.config.concat_mask
        rescale_scheduler_kwargs = kwargs.get('rescale_scheduler_kwargs',None)
        self.rescale_size_getter = get_ramp(rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.epoch_begin = opts.which_epoch
        self.total_epoch = opts.total_epoch
        self.pbar = tqdm(total=opts.total_epoch * self.train_set.dataset_size)
        self.global_step = self.epoch_begin * self.train_set.dataset_size
        self.generator_optim = make_optimizer(
            self.generator.parameters(),
            self.config.optimizers.generator.cls,
            **self.config.optimizers.generator.params)
        self.dis_optim = make_optimizer(
            self.discriminator.parameters(),
            self.config.optimizers.discriminator.cls,
            **self.config.optimizers.generator.params)
        self.fake_fakes_proba = fake_fakes_proba
        # if self.fake_fakes_proba > 1e-3:
        #     self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

    def train(self, **kwargs):
        if is_main_process():
            writer = SummaryWriter(logdir=self.config.log_save_dir)
        train_loader = self.train_set.get_dataloader()
        n = self.train_set.dataset_size
        if self.global_step > 0:
            self.pbar.update(self.global_step)
        global_iters = self.global_step
        data_iter = iter(train_loader)
        for epoch in range(self.epoch_begin, self.total_epoch):
            for curr_iter in range(n):
                batch = next(data_iter)
                batch = self.forward(batch)
                total_loss, metrics = self.discriminator_loss(batch)
                self.dis_optim.zero_grad()
                total_loss.backward()
                self.dis_optim.step()

                total_loss, gmetrics = self.generator_loss(batch)
                self.generator_optim.zero_grad()
                total_loss.backward()
                self.generator_optim.step()

                if is_main_process():
                    str_loss = [f"[{epoch}/{self.total_epoch} {curr_iter}/{n}]"]
                    loss_dict = {}
                    for k, v in metrics.items():
                        v = v.item()
                        loss_dict[k] = v
                        str_loss.append(f"{k}:{v:.3f}")
                    for k, v in gmetrics.items():
                        v = v.item()
                        loss_dict[k] = v
                        str_loss.append(f"f{k}:{v:.3f}")
                    self.pbar.desc = " ".join(str_loss)
                    self.pbar.update(1)
                    writer.add_scalars("loss", loss_dict, global_iters)
                    global_iters += 1
            self.show_middle(epoch, batch)
            if is_main_process():
                self.save_parameters(epoch)

    def show_middle(self, epoch, batch):
        files = batch["fname"]
        save_dir = f"{self.config['sample_save_dir']}/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        for k, tensor in batch.items():
            if k == "fname" or not isinstance(tensor, torch.Tensor):
                continue
            images = tensor.detach().cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = np.squeeze(images) * 255
            images = np.clip(images, 0, 255).astype(np.uint8)
            for file, img in zip(files, images):
                fname = os.path.splitext(os.path.basename(file))[0]
                if img.ndim == 3:
                    if img.shape[2] not in [1,3]:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f"{save_dir}/{fname}_{k}.jpg", img)

    def save_parameters(self, epoch):
        state = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict()
        }
        save_path = os.path.join(self.config.ckpt_save_dir, f"{epoch}.ckpt")
        torch.save(state, save_path)

    def forward(self, batch):
        if self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        img = batch['image'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        batch['image'] = img
        batch['mask'] = mask

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * img

        if self.fake_fakes_proba > 1e-3:
            if torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None else mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        # metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = self.config.losses.feature_matching.get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        # metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            # metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics
