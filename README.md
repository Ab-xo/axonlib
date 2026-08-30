# Axonlib

Axonlib is a lightweight, scratch-built neural network library implemented in Python with NumPy. It focuses on the core building blocks of a small deep learning workflow: layers, activations, loss functions, optimizers, and a simple sequential training loop.

## Features

- Dense affine layer implementation
- ReLU activation layer
- Mean squared error loss
- Stochastic gradient descent optimizer
- Minimal sequential model API for training and prediction
- Simple data loader for mini-batch training
- Unit tests covering core dense-layer behavior and a basic training example

## Project status

This repository is in its early development stage and is intended as a clean foundation for expanding into a more complete ML toolkit.

## Installation

Clone the repository and run from the project root:

```bash
git clone https://github.com/<your-username>/axonlib.git
cd axonlib
python -m pip install -r requirements.txt
```

## Quick example

```python
import numpy as np

from neura.layers import Dense
from neura.layers.activations import ReLU
from neura.losses.losses import MSELoss
from neura.model import Sequential
from neura.optimizers.optimizers import SGD

X = np.arange(-2.0, 3.0).reshape(-1, 1)
y = 3.0 * X - 1.0

model = Sequential([
    Dense(1, 1),
    ReLU(),
])

history = model.train(
    X,
    y,
    epochs=100,
    batch_size=5,
    loss_fn=MSELoss(),
    optimizer=SGD(lr=0.05),
    shuffle=False,
)

print(history[-1])
print(model.predict(X))
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

This project is currently unlicensed. Add a license file if you plan to make it public for broader use.
