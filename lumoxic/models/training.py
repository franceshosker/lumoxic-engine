"""Training loop for LNBE models."""

from typing import List, Callable, Optional
import numpy as np

from lumoxic.models.lnbe import LNBEModel, LNBEConfig


class Trainer:
    """Train LNBE models on binary stream data."""

    def __init__(
        self,
        model: Optional[LNBEModel] = None,
        config: Optional[LNBEConfig] = None,
    ):
        self.model = model or LNBEModel(config)
        self.history: List[float] = []

    def train(
        self,
        train_data: np.ndarray,
        train_targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """Train the model for the specified number of epochs."""
        n = len(train_data)

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = train_data[batch_idx]
                batch_y = train_targets[batch_idx]
                loss = self.model.train_step(batch_x, batch_y)
                epoch_loss += loss
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            self.history.append(avg_loss)

            if callback:
                callback(epoch, avg_loss)

        return self.history

    def evaluate(self, test_data: np.ndarray, test_targets: np.ndarray) -> dict:
        """Evaluate model on test data."""
        predictions = self.model.forward(test_data)
        mse = float(np.mean((predictions - test_targets) ** 2))
        mae = float(np.mean(np.abs(predictions - test_targets)))
        return {"mse": mse, "mae": mae, "samples": len(test_data)}

    def generate_training_data(
        self, n_samples: int = 1000, input_dim: int = 64, output_dim: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data from random binary streams."""
        rng = np.random.default_rng()
        inputs = rng.integers(0, 2, size=(n_samples, input_dim)).astype(np.float64)
        targets = np.zeros((n_samples, output_dim))
        for i in range(n_samples):
            targets[i] = inputs[i, :output_dim] * 0.8 + rng.normal(0, 0.1, output_dim)
        return inputs, targets
