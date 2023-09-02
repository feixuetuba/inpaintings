import os

from utils.directory import path_to_package
from utils.reflect import load_cls


def get_model(run_mode="test", **kwargs):
    curr_pkg = path_to_package(os.path.abspath(__file__))
    cls = None
    if run_mode == "test":
        cls = load_cls(f"{curr_pkg}.BigLamaModel", "BigLamaModel")
        return cls
    else:
        cls = load_cls(f"{curr_pkg}.traincodes.models.InpaintingModel", "InpaintingModel")
        return cls, cls.get_argparser()