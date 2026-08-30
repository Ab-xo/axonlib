import unittest

import numpy as np

from neura.layers import Dense


class DenseTests(unittest.TestCase):
    def test_forward_computes_affine_transform(self):
        layer = Dense(2, 3)
        layer.params["W"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.params["b"] = np.array([0.5, -0.5, 1.0])

        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        np.testing.assert_allclose(
            layer.forward(X),
            np.array([[9.5, 11.5, 16.0], [19.5, 25.5, 34.0]]),
        )

    def test_backward_computes_gradients_and_upstream_gradient(self):
        layer = Dense(2, 3)
        layer.params["W"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.params["b"] = np.zeros(3)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])

        layer.forward(X)
        upstream = layer.backward(grad)

        np.testing.assert_allclose(
            layer.grads["W"],
            np.array([[1.0, 6.0, 4.0], [2.0, 8.0, 6.0]]),
        )
        np.testing.assert_allclose(layer.grads["b"], np.array([1.0, 2.0, 2.0]))
        np.testing.assert_allclose(
            upstream,
            np.array([[4.0, 10.0], [7.0, 16.0]]),
        )


if __name__ == "__main__":
    unittest.main()
