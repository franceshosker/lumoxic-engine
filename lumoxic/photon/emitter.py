"""Photon generation and emission."""

from typing import List, Optional
import numpy as np

from lumoxic.core.types import Photon
from lumoxic.core.constants import DEFAULT_WAVELENGTH, MIN_WAVELENGTH, MAX_WAVELENGTH


class PhotonEmitter:
    """Generate photons with configurable wavelength, angle, and intensity."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def emit(
        self,
        wavelength: float = DEFAULT_WAVELENGTH,
        angle: float = 45.0,
        intensity: float = 1.0,
    ) -> Photon:
        """Emit a single photon with specified parameters."""
        angle = np.clip(angle, 0.0, 90.0)
        wavelength = np.clip(wavelength, MIN_WAVELENGTH, MAX_WAVELENGTH)
        rad = np.radians(angle)
        velocity = np.array([np.cos(rad), np.sin(rad)]) * intensity
        return Photon(
            wavelength=wavelength,
            angle=angle,
            intensity=intensity,
            position=np.array([0.0, 0.0]),
            velocity=velocity,
        )

    def emit_batch(
        self,
        count: int,
        wavelength: float = DEFAULT_WAVELENGTH,
        angle_range: tuple[float, float] = (5.0, 85.0),
        intensity: float = 1.0,
    ) -> List[Photon]:
        """Emit a batch of photons with random angles within range."""
        angles = self.rng.uniform(angle_range[0], angle_range[1], size=count)
        return [self.emit(wavelength, float(a), intensity) for a in angles]

    def emit_spectrum(
        self,
        count: int,
        wavelength_range: tuple[float, float] = (MIN_WAVELENGTH, MAX_WAVELENGTH),
        angle: float = 45.0,
    ) -> List[Photon]:
        """Emit photons across a wavelength spectrum."""
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], count)
        return [self.emit(float(w), angle) for w in wavelengths]
