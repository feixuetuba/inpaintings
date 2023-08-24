from utils.reflect import load_cls


class BaseModel:
    def __init__(self, *args, **kwargs):
        self._prepared = False

    def has_prepared(self):
        return self._prepared

    def prepare(self, *args, **kwargs):
        pass

    def forward(self, image, mask, **kwargs):
        pass



def get_algorithm(name, **kwargs):
    loader = load_cls(f"algorithms.{name}", "get_model")          #get_model 定义在__init__.py
    return loader(name, **kwargs)