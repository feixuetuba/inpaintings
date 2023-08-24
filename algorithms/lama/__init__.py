import os

from utils.directory import path_to_package
from utils.reflect import load_cls


def get_model(model_name="BigLamaModel", **kwargs):
    curr_pkg = path_to_package(os.path.abspath(__file__))
    model = load_cls(f"{curr_pkg}.BigLamaModel", "BigLamaModel")(**kwargs)
    return model