"""Photon trajectory calculation and ray tracing."""

from typing import List, Tuple
import numpy as np

from lumoxic.core.types import Surface


class TrajectoryCalculator:
    """Calculate photon trajectories through a bounce chamber."""

    def trace(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        surfaces: List[Surface],
    ) -> Tuple[int, np.ndarray, float]:
        """Trace a ray from position along velocity until it hits a surface.

        Returns (surface_index, hit_position, reflection_angle).
        Returns (-1, position, 0) if no intersection found.
        """
        min_t = float("inf")
        best_idx = -1
        best_pos = position.copy()
        best_angle = 0.0

        for i, surface in enumerate(surfaces):
            denom = np.dot(velocity, surface.normal)
            if abs(denom) < 1e-10:
                continue

            point_on_surface = np.array([surface.position, surface.position])
            t = np.dot(point_on_surface - position, surface.normal) / denom

            if t > 1e-6 and t < min_t:
                hit = position + velocity * t
                angle = np.arccos(
                    np.clip(abs(np.dot(velocity / np.linalg.norm(velocity), surface.normal)), 0, 1)
                )
                min_t = t
                best_idx = i
                best_pos = hit
                best_angle = angle

        return best_idx, best_pos, best_angle

    @staticmethod
    def reflect(velocity: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Reflect velocity vector off a surface with the given normal."""
        n = normal / np.linalg.norm(normal)
        return velocity - 2 * np.dot(velocity, n) * n

    @staticmethod
    def compute_path_length(positions: List[np.ndarray]) -> float:
        """Compute total path length through a series of positions."""
        if len(positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(positions)):
            total += np.linalg.norm(positions[i] - positions[i - 1])
        return total
