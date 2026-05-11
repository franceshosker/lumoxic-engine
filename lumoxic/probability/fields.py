"""Probability field mapping for photon distributions."""

import numpy as np


class ProbabilityField:
    """Map probability distributions of photon angles and binary outputs."""

    def __init__(self, resolution: int = 100):
        self.resolution = resolution

    def angle_distribution(
        self, n_samples: int = 10000, seed: int | None = None
    ) -> np.ndarray:
        """Generate angle probability distribution from random photon firings."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 90, n_samples)
        hist, _ = np.histogram(angles, bins=self.resolution, range=(0, 90), density=True)
        return hist

    def binary_probability(self, threshold: float = 45.0) -> tuple[float, float]:
        """Compute probability of 0 and 1 for uniform angle distribution."""
        p1 = (90.0 - threshold) / 90.0
        p0 = threshold / 90.0
        return p0, p1

    def conditional_probability(
        self, angles: list[float], threshold: float = 45.0
    ) -> dict:
        """Compute conditional bit probabilities given observed angles."""
        arr = np.array(angles)
        n = len(arr)
        if n == 0:
            return {"p0": 0.5, "p1": 0.5, "samples": 0}

        n1 = np.sum(arr >= threshold)
        n0 = n - n1
        return {
            "p0": float(n0 / n),
            "p1": float(n1 / n),
            "samples": n,
            "mean_angle": float(arr.mean()),
            "std_angle": float(arr.std()),
        }

    def monte_carlo_field(
        self,
        n_photons: int = 1000,
        grid_size: int = 50,
        threshold: float = 45.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a 2D probability field using Monte Carlo sampling.

        Each cell represents the probability of a photon at that position
        producing a high bit (1) based on its expected bounce angle.
        """
        rng = np.random.default_rng(seed)
        field = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        for _ in range(n_photons):
            x = rng.uniform(0, grid_size)
            y = rng.uniform(0, grid_size)
            angle = rng.uniform(0, 90)
            ix, iy = int(x) % grid_size, int(y) % grid_size
            field[iy, ix] += 1 if angle >= threshold else 0
            counts[iy, ix] += 1

        mask = counts > 0
        field[mask] /= counts[mask]
        return field
