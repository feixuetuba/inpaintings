import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

from algorithms import BaseModel
from algorithms.lama.networks import load_network
from utils.configure import Configure


class BigLamaModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._prepared = False
        self.device = kwargs.get("device", torch.device("cuda"))

    def prepare(self, *args, **kwargs):
        my_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_file = kwargs.get("cfg",my_dir + "/big-lama.yaml")
        cfg = Configure(cfg=cfg_file, TORCH_HOME=os.environ.get("TORCH_HOME", ""))
        self.network = load_network("generator",cfg)
        ckpt_file = cfg.generator.get("checkpoints", os.path.join(my_dir, "ckpts","big_lama_best.pth"))
        state_dict = torch.load( ckpt_file)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self._prepared = True

    def forward(self, img, mask, **kwargs):
        self.network.eval()
        h, w = img.shape[:2]

        stride = 8
        pt = pb = pl = pr = 0
        delta = h % stride
        if delta != 0:
            pb = stride - delta
        delta = w % stride
        if delta != 0:
            pr = stride - delta

        _mask = mask.copy()
        mask = cv2.copyMakeBorder(mask, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        mask = (mask.astype(float) >20)
        img = img.astype(np.float32) / 255.0
        mask = mask[..., None]
        masked_img = img * (1 - mask)

        with torch.no_grad():
            net_input = np.concatenate([masked_img[..., :3], mask], axis=2)
            net_input = np.transpose(net_input, (2,0,1))[None, ...]
            net_input = torch.from_numpy(net_input).float().to(self.device)
            predicted = self.network(net_input)[0].cpu().numpy()

            predicted = np.transpose(predicted, (1,2,0))
            result = masked_img[..., :3] + mask * predicted
            result = np.clip(result*255, 0, 255).astype(np.uint8)
        return result[:h, :w].astype(np.uint8)

