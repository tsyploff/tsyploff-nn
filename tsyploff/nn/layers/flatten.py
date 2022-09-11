import numpy as np
from ._base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, input_dim: int = 1):
        super(Flatten, self).__init__()
        self._name = "Flatten"
        self.input_dim = input_dim

    def forward(self, x: np.array) -> np.array:
        return x.flatten()

    def backward(self, chain: np.array, lr: float = .01) -> np.array:
        return chain.reshape(-1, self.input_dim)
