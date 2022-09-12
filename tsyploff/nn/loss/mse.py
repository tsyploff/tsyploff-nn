import numpy as np
from ._base_loss import BaseLoss


class MSE(BaseLoss):
    def __call__(self, y_true: np.array, y_pred: np.array) -> np.array:
        return (y_true - y_pred) **2 / 2

    @staticmethod
    def get_chain(y_true: np.array, y_pred: np.array) -> np.array:
        return y_pred - y_true
