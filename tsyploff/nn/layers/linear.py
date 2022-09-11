import numpy as np
from typing import Callable
from ._base_layer import BaseLayer


class Linear(BaseLayer):
    def __init__(
            self,
            input_shape: int,
            output_shape: int,
            init: Callable[[int, int], np.array],
            bias: bool = True
    ):
        super().__init__()
        self._name = "Linear"
        self.bias = bias
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = init(input_shape, output_shape)  # (m, k)
        self.bias_weights = init(1, output_shape)  # (1, k)

    def __repr__(self):
        return f"Linear(input_shape={self.input_shape}, output_shape={self.output_shape})"

    def forward(self, x: np.array) -> np.array:
        if self.bias:
            return x.dot(self.weights) + self.bias_weights
        return x.dot(self.weights)

    def backward(self, chain: np.array) -> np.array:
        return chain.dot(self.weights.T)

    def update_weights(self, chain: np.array, lr: float = .01) -> None:
        self.weights -= lr * self._last_input.T.dot(chain) / chain.shape[0]
        self.bias_weights -= lr * chain.mean(axis=0)
