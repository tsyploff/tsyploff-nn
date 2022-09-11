import numpy as np
from ._base_layer import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self._name = "Sigmoid"

    def forward(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, chain: np.array, lr: float = .01) -> np.array:
        sigma = self.forward(self._last_input)
        return chain * sigma * (1 - sigma)
