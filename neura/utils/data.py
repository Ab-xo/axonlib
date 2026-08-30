import numpy as np


class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.features = np.asarray(X)
        self.targets = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield self.features[batch_indices], self.targets[batch_indices]