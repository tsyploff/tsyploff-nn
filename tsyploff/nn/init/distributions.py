import numpy as np
from typing import Callable


def standard_uniform(input_dim: int, output_dim: int) -> np.array:
    return np.random.uniform(0, 1, (input_dim, output_dim))


def normal(mean: float, std: float) -> Callable[[int, int], np.array]:
    def wrapper(input_dim: int, output_dim: int) -> np.array:
        return np.random.normal(mean, std, (input_dim, output_dim))
    return wrapper


def standard_normal(input_dim: int, output_dim: int) -> np.array:
    return normal(0, 1)(input_dim, output_dim)
