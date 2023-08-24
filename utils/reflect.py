import importlib
import os
import sys


def load_cls(pkg, cls):
    package = importlib.import_module(pkg)
    return getattr(package, cls)


def load_curr_cls(file, module_name):
    fname = os.path.splitext(file)[0]
    curr_module =  sys.modules[module_name]
    return getattr(curr_module, fname)