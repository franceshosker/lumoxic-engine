"""Unified Client API for the Lumoxic photon computing engine."""

from typing import List, Optional
import numpy as np

from lumoxic.core.types import Photon, SimulationResult, BinaryStream
from lumoxic.bounce.engine import BinaryBounceEngine
from lumoxic.photon.emitter import PhotonEmitter
from lumoxic.photon.processor import PhotonProcessor
from lumoxic.binary.encoder import AngleEncoder
from lumoxic.binary.patterns import PatternAnalyzer
from lumoxic.models.lnbe import LNBEModel, LNBEConfig
from lumoxic.models.inference import InferenceEngine
from lumoxic.optics.wavelength import WavelengthAnalyzer


class Client:
    """Unified interface to the Lumoxic photon computing engine.

    Example:
        from lumoxic import Client

        client = Client(threshold=45.0, seed=42)
        result = client.process(wavelength=450, angle=37, bounces=8)
        print(result.binary_stream.as_string)
    """

    def __init__(
        self,
        threshold: float = 45.0,
        seed: Optional[int] = None,
        model_config: Optional[LNBEConfig] = None,
    ):
        self.engine = BinaryBounceEngine(threshold=threshold, seed=seed)
        self.emitter = PhotonEmitter(seed=seed)
        self.processor = PhotonProcessor(self.engine)
        self.encoder = AngleEncoder(threshold=threshold)
        self.analyzer = PatternAnalyzer()
        self.wavelength = WavelengthAnalyzer()
        self._model: Optional[LNBEModel] = None
        self._model_config = model_config
        self._inference: Optional[InferenceEngine] = None

    def process(
        self,
        wavelength: float = 450.0,
        angle: float = 45.0,
        bounces: int = 8,
    ) -> SimulationResult:
        """Process a single photon through the binary bounce engine."""
        photon = self.emitter.emit(wavelength=wavelength, angle=angle)
        return self.processor.process(photon, max_bounces=bounces)

    def simulate(
        self,
        photon_count: int = 256,
        wavelength: float = 450.0,
        max_bounces: int = 8,
    ) -> SimulationResult:
        """Run a batch simulation with multiple photons."""
        return self.engine.simulate(
            photon_count=photon_count,
            wavelength=wavelength,
            max_bounces=max_bounces,
        )

    def encode(self, angles: List[float]) -> str:
        """Encode a list of angles to a binary string."""
        return self.encoder.encode_to_string(angles)

    def analyze(self, bits: List[int]) -> dict:
        """Analyze a binary stream for patterns and entropy."""
        return {
            "length": len(bits),
            "entropy": self.analyzer.entropy(bits),
            "frequency": self.analyzer.frequency(bits),
            "autocorrelation": self.analyzer.autocorrelation(bits, max_lag=5),
            "top_patterns": dict(list(self.analyzer.find_repeating(bits).items())[:5]),
        }

    def infer(self, binary_input: List[int]) -> np.ndarray:
        """Run model inference on binary data."""
        if self._inference is None:
            self._model = LNBEModel(self._model_config)
            self._inference = InferenceEngine(self._model)
        return self._inference.infer(binary_input)

    def train(
        self, train_data: np.ndarray, targets: np.ndarray, epochs: int = 100
    ) -> List[float]:
        """Train the LNBE model."""
        from lumoxic.models.training import Trainer
        if self._model is None:
            self._model = LNBEModel(self._model_config)
            self._inference = InferenceEngine(self._model)
        trainer = Trainer(self._model)
        return trainer.train(train_data, targets, epochs=epochs)
