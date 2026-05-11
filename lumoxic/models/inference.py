"""Model inference on binary streams."""

from typing import List
import numpy as np

from lumoxic.models.lnbe import LNBEModel


class InferenceEngine:
    """Run inference on binary streams using LNBE models."""

    def __init__(self, model: LNBEModel | None = None):
        self.model = model or LNBEModel()

    def infer(self, binary_input: List[int]) -> np.ndarray:
        """Run inference on a binary stream."""
        return self.model.predict(binary_input)

    def classify(self, binary_input: List[int], num_classes: int = 10) -> int:
        """Classify a binary stream into one of num_classes categories."""
        output = self.infer(binary_input)
        scores = output[:num_classes]
        return int(np.argmax(scores))

    def batch_infer(self, inputs: List[List[int]]) -> List[np.ndarray]:
        """Run inference on a batch of binary streams."""
        return [self.infer(inp) for inp in inputs]

    def confidence(self, binary_input: List[int]) -> float:
        """Get confidence score for the prediction."""
        output = self.infer(binary_input)
        exp = np.exp(output - np.max(output))
        softmax = exp / exp.sum()
        return float(np.max(softmax))
