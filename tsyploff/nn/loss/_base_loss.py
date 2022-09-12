import numpy as np
from abc import ABCMeta, abstractmethod


class BaseLoss(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array) -> np.array:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def get_chain(y_true: np.array, y_pred: np.array) -> np.array:
        return NotImplemented
