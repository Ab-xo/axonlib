import numpy as np


def accuracy(y_pred, y_true):
    predicted_classes = (
        np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred >= 0.5)
    )
    true_classes = (
        np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    )
    return np.mean(predicted_classes == true_classes)
