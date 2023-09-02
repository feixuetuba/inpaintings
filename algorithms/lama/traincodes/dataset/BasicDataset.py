import glob
import os

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


class BasicDataset(Dataset):
    def __init__(self, cfg):
        self.samples = []

    def load(self, cfg):
        config = cfg.dataset
        self.config = config
        indir = config.train.dir_path
        dirs = [f"{indir}/images"]
        while len(dirs) != 0:
            dpath = dirs.pop()
            for f in os.listdir(dpath):
                fpath = f"{dpath}/{f}"
                if os.path.isfile(fpath):
                    fname, suffix = os.path.splitext(f)
                    mask_f = fpath.replace("images", "masks").replace(suffix, "_mask.png")
                    if not os.path.isfile(mask_f):
                        print("missing:", mask_f)
                        continue
                    self.samples.append((fpath, mask_f))
                elif os.path.isdir(fpath):
                    dirs.append(fpath)

        self.out_size = config.train.out_size
        return {"log":f"input_dir:{indir}\ncount:{len(self.samples)}"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        fpath, mpath = self.samples[item]
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size)).astype(float) / 255.0
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))[..., None].astype(float) / 255.0
        return dict(image=np.transpose(img, (2,0,1)),
                    mask=np.transpose(mask, (2,0,1)),
                    fname=os.path.basename(fpath))