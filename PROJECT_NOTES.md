# Axonlib Project Notes

## Purpose

Axonlib is a scratch-built Python neural-network library intended to grow into a practical, extensible toolkit for training models and solving optimization problems. The project is currently establishing the low-level building blocks rather than presenting a complete training API.

## Current structure

- `neura.layers` contains the first layer abstractions and implementations.
- `neura.losses` provides mean squared error.
- `neura.optimizers` provides stochastic gradient descent.
- `neura.utils` provides mini-batch loading and accuracy calculation.
- `tests` contains dense-layer tests and an end-to-end training test.
- NumPy is the numerical foundation used by the implemented layers.

## Layer design

`Layer` in `neura/layers/base.py` defines the common contract:

- `params` stores trainable arrays such as weights and biases.
- `grads` stores derivatives for those arrays after backpropagation.
- `forward(X)` computes a layer's output.
- `backward(grad)` receives the gradient from the next layer and returns the gradient for the previous layer.

`Dense` in `neura/layers/dense.py` implements an affine layer:

```text
output = X @ W + b
```

It initializes `W` with small random values and `b` with zeros. During backpropagation it caches the input and computes `dW`, `db`, and the upstream input gradient using NumPy matrix operations.

`ReLU` in `neura/layers/activations.py` is the first activation layer. It caches a positive-value mask during the forward pass and uses that mask to pass gradients only through positive inputs.

## Current implementation status

`Sequential` composes layers, runs forward and backward passes, trains over mini-batches, updates every layer with an optimizer, predicts outputs, and returns one mean loss per epoch.

`DataLoader` validates sample and batch-size contracts, optionally shuffles sample indices, and yields the final partial batch when needed. `MSELoss` supplies the loss and gradient, while `SGD` applies gradients stored by each layer.

## Current constraints and open work

- Input validation is minimal: shape, dtype, and forward-before-backward requirements are not explicitly checked.
- Random initialization is not seeded or configurable through a shared random generator.
- The package is not yet installable as a formal distribution; tests currently rely on running from the repository root.
- Public conventions for gradient shapes should be documented consistently. Dense bias parameters remain shape `(1, output_size)`, while their stored gradients are shape `(output_size,)`; NumPy broadcasting makes the optimizer update valid, but a uniform convention would be clearer.

## Verification snapshot

From the `axonlib` repository root:

- The dense forward and backward tests pass.
- The end-to-end linear-regression training test passes.
- Python compilation passes for the package and tests.
- `ReLU` is publicly exported from `neura.layers`.

## Suggested growth direction

1. Decide and document tensor and gradient shape conventions.
2. Add activation tests, including ReLU forward and backward behavior.
3. Add gradient-checking utilities and numerical gradient tests.
4. Add a classification loss, a stable softmax layer, and classification training tests.
5. Add validation/evaluation APIs, model serialization, and richer optimizer state such as momentum or Adam.
6. Add packaging metadata once the API stabilizes.
