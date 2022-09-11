import numpy as np
from typing import List
from .layers import BaseLayer


class Sequental:
    def __init__(self, layers: List[BaseLayer], training: bool = False):
        self.layers = layers
        self.training = training
        print(self.__repr__())

    def __repr__(self):
        return "Sequental(\n    " + ",\n    ".join([layer.__str__() for layer in self.layers]) + "\n)"

    def __str__(self):
        return self.__repr__()

    def __call__(self, x: np.array) -> np.array:
        return self.forward(x)

    def add(self, layer: BaseLayer) -> 'Sequental':
        self.layers.append(layer)
        return self

    def set_training(self, training: bool = False) -> None:
        self.training = training

    def forward(self, x: np.array) -> np.array:
        current = x
        for layer in self.layers:
            if self.training:
                layer.remember_input(current)
            current = layer.forward(current)
        return current

    def backward(self, chain: np.array, lr: float = .01) -> None:
        current = chain
        for layer in self.layers[::-1]:
            layer.update_weights(current, lr=lr)
            current = layer.backward(current)
