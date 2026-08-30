import numpy as np

class Layer:
    """Base class for all layer"""
    def __init__(self):
        self.params = {} # trainable parameters(wights,biases)
        self.grads = {} # gradients for parameters

    def forward(self, X:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def backward(self,grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError