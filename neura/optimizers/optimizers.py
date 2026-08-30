import numpy as np

class SGD:
    def __init__(self, lr =0.01):
        self.lr = lr

    def update(self,layer):
        for param_name in layer.params:
            layer.params[param_name] -= self.lr * layer.grads[param_name]
