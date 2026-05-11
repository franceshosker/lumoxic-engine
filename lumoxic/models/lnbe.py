"""LNBE (Light-to-Neural Binary Engine) base model."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class LNBEConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    num_layers: int = 3
    learning_rate: float = 0.001
    activation: str = "relu"


class LNBEModel:
    """Light-to-Neural Binary Engine — neural network for photon-derived binary data.

    A simple feedforward network that processes binary streams from the bounce
    engine and produces predictions/classifications.
    """

    def __init__(self, config: Optional[LNBEConfig] = None, seed: Optional[int] = None):
        self.config = config or LNBEConfig()
        self.rng = np.random.default_rng(seed)
        self._init_weights()
        self._trained = False
        self._epoch = 0

    def _init_weights(self) -> None:
        dims = [self.config.input_dim] + [self.config.hidden_dim] * self.config.num_layers + [self.config.output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(self.rng.normal(0, scale, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = self._activate(x)
        return x

    def predict(self, binary_input: List[int]) -> np.ndarray:
        """Predict from a binary input stream."""
        x = np.array(binary_input, dtype=np.float64)
        if len(x) < self.config.input_dim:
            x = np.pad(x, (0, self.config.input_dim - len(x)))
        elif len(x) > self.config.input_dim:
            x = x[: self.config.input_dim]
        return self.forward(x.reshape(1, -1)).flatten()

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Single training step. Returns loss."""
        predictions = self.forward(inputs)
        loss = np.mean((predictions - targets) ** 2)

        # Simplified gradient descent on output layer
        error = predictions - targets
        grad = error / len(inputs)
        self.weights[-1] -= self.config.learning_rate * (
            self._activate(inputs @ self.weights[-2] + self.biases[-2]).T @ grad
            if len(self.weights) > 1
            else inputs.T @ grad
        )
        self.biases[-1] -= self.config.learning_rate * grad.mean(axis=0)

        self._epoch += 1
        self._trained = True
        return float(loss)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def parameter_count(self) -> int:
        total = sum(w.size for w in self.weights)
        total += sum(b.size for b in self.biases)
        return total

    def summary(self) -> str:
        return (
            f"LNBEModel(layers={self.config.num_layers}, "
            f"params={self.parameter_count:,}, "
            f"trained={self._trained}, epochs={self._epoch})"
        )
