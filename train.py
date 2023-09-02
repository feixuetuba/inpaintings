import logging
import os
import time
from argparse import ArgumentParser

import torch

from algorithms import get_algorithm
from utils.ddp import launch


def get_opts():
    parser = ArgumentParser()
    parser.add_argument("cfg", default="checkpoints/example/example.yaml", help="path to config file")
    parser.add_argument("--version", help="version of this experiment")
    parser.add_argument("--which_epoch", type=int, default=-1, help="which epoch to load")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrain checkpoint")
    return parser.parse_args()

def main(cls, opts, save_dir, cudas, use_ddp):
    model = cls()
    model.prepare(opts, save_dir, cudas, use_ddp)
    logging.info("model [%s] was created" % (model.__class__.__name__))

    model.train()


if __name__=="__main__":
    algorithm = "lama"

    cls, parser = get_algorithm(algorithm, run_mode="train")
    opts = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    local_time = time.localtime()

    full_time = time.strftime("%Y/%m/%d %H:%M:%S", local_time)
    date_only = time.strftime("%Y_%m_%d", local_time)

    save_dir = "experiments"
    exp_root = f"{save_dir}/{date_only}"
    exp_name = os.path.splitext(os.path.basename(opts.cfg))[0]

    ncuda = torch.cuda.device_count()
    cuda_avaliable = list(range(ncuda))

    if len(cuda_avaliable) > 1:
        launch(
            main,
            num_gpus_per_machine=ncuda,  # 进程数目
            machine_rank=0,
            dist_url="auto",
            args=(cls, opts, save_dir, cuda_avaliable, True )
        )
    else:
        main(cls, opts, save_dir, cuda_avaliable, False )
