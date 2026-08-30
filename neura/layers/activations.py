import numpy as np
from.base import Layer

class ReLU(Layer):
    def forward(self,X):
        self.mask =(X > 0)
        return X * self.mask
    def backward(self,grad):
        return grad * self.mask