import os.path

from utils.directory import path_to_package
from utils.reflect import load_cls


def load_network(net_type,config, **kwargs):
    curr_pkg = path_to_package(os.path.abspath(__file__))
    net_cfg = config[net_type]
    net_name = net_cfg["cls"]
    pkg = net_cfg.get("package", net_name)
    network = load_cls(f"{curr_pkg}.{pkg}", net_name)
    params = network.from_config(net_cfg, **kwargs)
    return network(**params)