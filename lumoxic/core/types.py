"""Type definitions for the Lumoxic photon computing engine."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Photon:
    wavelength: float  # nanometers
    angle: float  # degrees (incidence angle)
    intensity: float = 1.0
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))

    @property
    def frequency(self) -> float:
        return 3e17 / self.wavelength  # Hz

    @property
    def energy(self) -> float:
        return 6.626e-34 * self.frequency  # Joules


@dataclass
class Surface:
    name: str
    reflection_coefficient: float  # 0.0 to 1.0
    position: float  # y-coordinate of the surface
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    material: str = "glass"


@dataclass
class Bounce:
    surface_index: int
    incidence_angle: float  # degrees
    reflection_angle: float  # degrees
    bit: int  # 0 or 1
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    energy_loss: float = 0.0


@dataclass
class BinaryStream:
    bits: List[int] = field(default_factory=list)
    source_angles: List[float] = field(default_factory=list)
    threshold: float = 45.0

    @property
    def as_string(self) -> str:
        return "".join(str(b) for b in self.bits)

    @property
    def length(self) -> int:
        return len(self.bits)

    @property
    def entropy(self) -> float:
        if not self.bits:
            return 0.0
        p1 = sum(self.bits) / len(self.bits)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0.0
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    def append(self, angle: float) -> int:
        bit = 1 if angle >= self.threshold else 0
        self.bits.append(bit)
        self.source_angles.append(angle)
        return bit


@dataclass
class SimulationResult:
    binary_stream: BinaryStream
    bounces: List[Bounce]
    total_energy_loss: float = 0.0
    photon_count: int = 0
    processing_time_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        if not self.bounces:
            return 0.0
        valid = sum(1 for b in self.bounces if b.energy_loss < 0.1)
        return valid / len(self.bounces)
