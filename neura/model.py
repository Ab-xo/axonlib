import numpy as np

from .utils.data import DataLoader


class Sequential:
    def __init__(self, layers=None):
        self.layers = list(layers) if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self,grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs, batch_size, loss_fn, optimizer, shuffle=True):
        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_X, batch_y in DataLoader(X, y, batch_size, shuffle):
                predictions = self.forward(batch_X)
                epoch_loss += loss_fn.forward(predictions, batch_y)
                self.backward(loss_fn.backward(predictions, batch_y))
                for layer in self.layers:
                    optimizer.update(layer)
                batch_count += 1
            history.append(epoch_loss / batch_count)
        return np.asarray(history)