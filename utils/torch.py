from torch import  nn


def get_activation(kind='tanh', **kwargs):
    kind = kind.lower()
    if kind == 'tanh':
        return nn.Tanh()
    elif kind == 'sigmoid':
        return nn.Sigmoid()
    elif kind == 'relu':
        return nn.ReLU()
    elif kind == "":
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')