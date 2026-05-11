"""Binary Bounce Engine — core photon-to-binary simulation."""

import time
from typing import List, Optional
import numpy as np

from lumoxic.core.types import Photon, Bounce, BinaryStream, Surface, SimulationResult
from lumoxic.core.constants import DEFAULT_THRESHOLD, SURFACE_MATERIALS
from lumoxic.bounce.trajectory import TrajectoryCalculator


class BinaryBounceEngine:
    """Core engine that converts photon bounces into binary data.

    Photons are fired into a chamber with reflective surfaces. Each bounce
    produces a reflection angle. Angles >= threshold become 1, below become 0.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        surfaces: Optional[List[Surface]] = None,
        seed: Optional[int] = None,
    ):
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.trajectory = TrajectoryCalculator()

        if surfaces is None:
            self.surfaces = [
                Surface("bottom", 0.92, 0.0, np.array([0.0, 1.0]), "glass"),
                Surface("top", 0.92, 10.0, np.array([0.0, -1.0]), "glass"),
                Surface("right", 0.95, 20.0, np.array([-1.0, 0.0]), "mirror"),
            ]
        else:
            self.surfaces = surfaces

    def fire_photon(self, photon: Photon, max_bounces: int = 8) -> SimulationResult:
        """Fire a single photon and collect binary output from bounces."""
        start = time.perf_counter()
        stream = BinaryStream(threshold=self.threshold)
        bounces: List[Bounce] = []
        total_loss = 0.0

        pos = photon.position.copy()
        angle_rad = np.radians(photon.angle)
        velocity = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * photon.intensity

        for i in range(max_bounces):
            surface_idx, hit_pos, reflection_angle = self.trajectory.trace(
                pos, velocity, self.surfaces
            )
            if surface_idx < 0:
                break

            surface = self.surfaces[surface_idx]
            energy_loss = 1.0 - surface.reflection_coefficient
            total_loss += energy_loss

            angle_deg = np.degrees(abs(reflection_angle))
            angle_deg = min(90.0, max(0.0, angle_deg))

            bit = stream.append(angle_deg)

            bounce = Bounce(
                surface_index=surface_idx,
                incidence_angle=photon.angle if i == 0 else bounces[-1].reflection_angle,
                reflection_angle=angle_deg,
                bit=bit,
                position=hit_pos.copy(),
                energy_loss=energy_loss,
            )
            bounces.append(bounce)

            pos = hit_pos
            velocity = self.trajectory.reflect(velocity, surface.normal)
            velocity *= surface.reflection_coefficient

        elapsed = (time.perf_counter() - start) * 1000

        return SimulationResult(
            binary_stream=stream,
            bounces=bounces,
            total_energy_loss=total_loss,
            photon_count=1,
            processing_time_ms=elapsed,
        )

    def simulate(
        self, photon_count: int = 256, wavelength: float = 450.0, max_bounces: int = 8
    ) -> SimulationResult:
        """Run a batch simulation with multiple photons."""
        start = time.perf_counter()
        all_stream = BinaryStream(threshold=self.threshold)
        all_bounces: List[Bounce] = []
        total_loss = 0.0

        for _ in range(photon_count):
            angle = self.rng.uniform(5, 85)
            photon = Photon(wavelength=wavelength, angle=angle)
            result = self.fire_photon(photon, max_bounces=max_bounces)

            all_stream.bits.extend(result.binary_stream.bits)
            all_stream.source_angles.extend(result.binary_stream.source_angles)
            all_bounces.extend(result.bounces)
            total_loss += result.total_energy_loss

        elapsed = (time.perf_counter() - start) * 1000

        return SimulationResult(
            binary_stream=all_stream,
            bounces=all_bounces,
            total_energy_loss=total_loss,
            photon_count=photon_count,
            processing_time_ms=elapsed,
        )
