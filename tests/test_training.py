import unittest

import numpy as np

from neura.layers import Dense
from neura.losses.losses import MSELoss
from neura.model import Sequential
from neura.optimizers.optimizers import SGD


class TrainingTests(unittest.TestCase):
    def test_sequential_model_learns_linear_relationship(self):
        np.random.seed(7)
        features = np.arange(-2.0, 3.0).reshape(-1, 1)
        targets = 3.0 * features - 1.0
        model = Sequential([Dense(1, 1)])

        history = model.train(
            features,
            targets,
            epochs=300,
            batch_size=5,
            loss_fn=MSELoss(),
            optimizer=SGD(lr=0.05),
            shuffle=False,
        )

        self.assertLess(history[-1], history[0])
        np.testing.assert_allclose(model.predict(features), targets, atol=1e-2)


if __name__ == "__main__":
    unittest.main()