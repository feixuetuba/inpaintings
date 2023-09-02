import logging
import os

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import  BatchSampler
import utils.ddp as comm
from algorithms.lama.traincodes.dataset.samplers import WeightedSampler, TrainingSampler, InferenceSampler
from utils.directory import path_to_package

from utils.reflect import load_cls


def build_dataset(cfg, set_type="train"):
    data_cfg = cfg.dataset[set_type]
    curr_pkg = path_to_package(os.path.abspath(__file__))
    cls = data_cfg["cls"]
    package = data_cfg.get('package', cls)
    obj = load_cls(f"{curr_pkg}.{package}", cls)(data_cfg)
    return obj


class BuildDataloader:
    def __init__(self, cfg, is_train=True, collate_fn=None):
        dataset = build_dataset(cfg, "train" if is_train else "val")
        ds_info = dataset.load(cfg)
        if comm.is_main_process():
            save_file = os.path.join(cfg.txt_logs_dir, "ds_train.txt" if is_train else "ds_test.txt")
            with open(save_file, "w") as fd:
                fd.write(ds_info["log"])

        if is_train:
            # 训练 train
            # dataset = eval(cfg.dataset.DATASET_NAME)(cfg, is_train)

            self.dataset_size = len(dataset)
            # 多卡
            print("Build_loader, GPU",cfg.gpu_ids )
            if len(cfg.cudas) > 1 and cfg.use_ddp:
                batch_size_per_gpu = cfg.dataset.train.batch_size_per_cuda # // len(cfg.SOLVERS.GPU_IDS)
                if "weights" in ds_info:
                    sampler = WeightedSampler(
                        weights = ds_info["weights"],
                        num_samples = len(dataset),
                        seed=cfg.dataset.get("seed", None),
                        replacement=ds_info.get("replacement", True),
                    )
                else:
                    sampler = TrainingSampler(len(dataset), shuffle=cfg.dataset.train.shuffle,seed=cfg.dataset.get("seed", None))
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.dataset.train.drop_last)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.dataset.train.num_workers,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
            else:
                logging.info("Single GPU")
                batch_size_per_gpu = cfg.dataset.train.batch_size_per_cuda
                sampler = TrainingSampler(len(dataset), shuffle=cfg.dataset.train.shuffle)
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.dataset.train.drop_last)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.dataset.train.num_workers,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
        else:
            # dataset = eval(cfg.dataset.DATASET_NAME)(cfg, is_train)
            self.dataset_size = len(dataset)
            # 多卡
            if len(cfg.gpu_ids) > 1 and cfg.use_ddp:
                batch_size_per_gpu = cfg.dataset.test.batch_size_per_cuda // len(cfg.gpu_ids)
                sampler = InferenceSampler(len(dataset))
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.dataset.test.drop_last)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.dataset.test.num_workers,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
            else:
                # 单卡
                self.dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.dataset.test.batch_size_per_cuda,
                    shuffle=cfg.dataset.test.shuffle,
                    num_workers=cfg.dataset.test.num_workers,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn,
                    drop_last=cfg.dataset.test.drop_last
                )

    def get_dataloader(self):
        return self.dataloader

    def get_dataset_size(self):
        return self.dataset_size


