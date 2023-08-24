import re

import yaml


def load_config_file(yaml_file, **kwargs):
    with open(yaml_file, "r") as fd:
        data = fd.read()

    g=re.findall('[\"|\']{3}\b*\n*[\w\W]*\b*\n*[\"|\']{3}', data)

    data = re.sub('[\"|\']{3}\b*\n*[\w\W]*\b*\n*[\"|\']{3}\b*\n*', '', data)
    cfg = yaml.safe_load(data)
    comment = ""
    if len(g) > 0:
        comment = g[0]

    return cfg, comment


class Configure(dict):
    def __init__(self, cfg=None, **kwargs):
        comment = ""
        ROOT = kwargs.get("ROOT", self)
        if cfg is not None:
            if isinstance(cfg, str):
                if cfg.endswith(".yaml"):
                    cfg, comment = load_config_file(cfg, **kwargs)
                else:
                    raise ValueError(f"Unsepported format:{cfg}")
        else:
            cfg = {}
        if len(comment.strip()) != 0:
            setattr(self, "COMMENT", comment)
        for k, v in cfg.items():
            if k == "parent_config":
                parent_cfg = Configure(v)
                for k, v in parent_cfg.items():
                    setattr(self, k , v)
            else:
                if isinstance(v, str) and re.match('\$?\{.*\}', v) is not None:
                    if v in kwargs:
                        v = kwargs[v]
                    else:
                        if hasattr(self, v):
                            v = getattr(self, v)
                        else:
                            curr = ROOT
                            v = re.sub('\$?\{|\}','',v)
                            for sub in v.split("."):
                                if hasattr(curr, sub):
                                    curr = getattr(curr, sub)
                                elif hasattr(self, sub):
                                    curr = getattr(self, sub)
                                else:
                                    assert False, f"value {v} for {k} no found, stop at >{sub}<"
                            v = curr
                elif isinstance(v, dict):
                    v = Configure(v, ROOT=ROOT)
                setattr(self, k, v)

        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                v = getattr(self, k)
                setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Configure, self).__setattr__(name, value)
        super(Configure, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Configure, self).pop(k, d)