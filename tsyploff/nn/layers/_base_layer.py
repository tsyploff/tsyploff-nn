import numpy as np
from abc import ABCMeta, abstractmethod


class BaseLayer(metaclass=ABCMeta):
    def __init__(self):
        self._last_input = None
        self._name = "Layer"

    def __repr__(self):
        return f"{self._name}()"

    def __str__(self):
        return self.__repr__()

    def remember_input(self, x: np.array) -> None:
        self._last_input = x.copy()

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Evaluate layer's forward function.

        :param x: array-like, x.shape = (n, m) where n is batch_size and m is input dimension
        :return: array-like of shape (n,) of (n, k) where k is output dimension
        """
        return NotImplemented

    @abstractmethod
    def backward(self, chain: np.array) -> np.array:
        """
        Finds derivative of layer's function by x.

        :param chain: array-like of shape (k,) where k is output dimension
        :return: array-like of shape (m,) where m is input dimension
        """
        return NotImplemented

    def update_weights(self, chain: np.array, lr: float = .01) -> None:
        """
        Common method for update weights on layer
        :param lr: learning rate
        :param chain: array-like of shape (n, k) where k is output dimension
        :return:
        """
        pass
