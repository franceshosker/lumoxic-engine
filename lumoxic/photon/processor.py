"""Photon-to-binary conversion pipeline."""

from typing import List
import numpy as np

from lumoxic.core.types import Photon, BinaryStream, SimulationResult
from lumoxic.bounce.engine import BinaryBounceEngine


class PhotonProcessor:
    """High-level photon processing pipeline.

    Handles photon intake, bounce simulation, and binary output assembly.
    """

    def __init__(self, engine: BinaryBounceEngine | None = None):
        self.engine = engine or BinaryBounceEngine()
        self._processed_count = 0

    def process(self, photon: Photon, max_bounces: int = 8) -> SimulationResult:
        """Process a single photon through the bounce engine."""
        result = self.engine.fire_photon(photon, max_bounces)
        self._processed_count += 1
        return result

    def process_batch(
        self, photons: List[Photon], max_bounces: int = 8
    ) -> SimulationResult:
        """Process a batch of photons and merge binary output."""
        merged_stream = BinaryStream(threshold=self.engine.threshold)
        all_bounces = []
        total_loss = 0.0
        total_time = 0.0

        for photon in photons:
            result = self.process(photon, max_bounces)
            merged_stream.bits.extend(result.binary_stream.bits)
            merged_stream.source_angles.extend(result.binary_stream.source_angles)
            all_bounces.extend(result.bounces)
            total_loss += result.total_energy_loss
            total_time += result.processing_time_ms

        return SimulationResult(
            binary_stream=merged_stream,
            bounces=all_bounces,
            total_energy_loss=total_loss,
            photon_count=len(photons),
            processing_time_ms=total_time,
        )

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def reset(self) -> None:
        self._processed_count = 0
