import numpy as np


class BinaryCrossEntropy:
    def __call__(self, y_true: np.array, y_pred: np.array) -> np.array:
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    @staticmethod
    def get_chain(y_true: np.array, y_pred: np.array) -> np.array:
        return (1 - y_true) / (1 - y_pred) - y_true / y_pred
