import os.path

from utils.directory import path_to_package
from utils.reflect import load_cls


def load_network(net_name, config, **kwargs):
    curr_pkg = path_to_package(os.path.abspath(__file__))
    pkg = config.get("package", net_name)
    network = load_cls(f"{curr_pkg}.{pkg}", net_name)
    params = network.from_config(config, **kwargs)
    return network(**params)