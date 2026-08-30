import numpy as np
from .base import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, init_scale: float = 0.01):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params['W'] = np.random.randn(
            input_size, output_size) * init_scale
        self.params['b'] = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads['W'] = self.X.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['W'].T
