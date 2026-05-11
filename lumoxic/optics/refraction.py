"""Snell's law and refraction calculations."""

import numpy as np


class RefractionCalculator:
    """Calculate refraction and reflection angles using Snell's law."""

    @staticmethod
    def snells_law(angle_i: float, n1: float, n2: float) -> float | None:
        """Apply Snell's law. Returns refracted angle or None for total internal reflection."""
        sin_r = (n1 / n2) * np.sin(np.radians(angle_i))
        if abs(sin_r) > 1.0:
            return None  # Total internal reflection
        return float(np.degrees(np.arcsin(sin_r)))

    @staticmethod
    def critical_angle(n1: float, n2: float) -> float | None:
        """Compute critical angle for total internal reflection."""
        if n1 <= n2:
            return None  # No total internal reflection possible
        return float(np.degrees(np.arcsin(n2 / n1)))

    @staticmethod
    def fresnel_reflectance(angle_i: float, n1: float, n2: float) -> float:
        """Compute Fresnel reflectance (average of s and p polarization)."""
        angle_r = RefractionCalculator.snells_law(angle_i, n1, n2)
        if angle_r is None:
            return 1.0  # Total internal reflection

        cos_i = np.cos(np.radians(angle_i))
        cos_r = np.cos(np.radians(angle_r))

        rs = ((n1 * cos_i - n2 * cos_r) / (n1 * cos_i + n2 * cos_r)) ** 2
        rp = ((n1 * cos_r - n2 * cos_i) / (n1 * cos_r + n2 * cos_i)) ** 2

        return float((rs + rp) / 2)

    @staticmethod
    def brewster_angle(n1: float, n2: float) -> float:
        """Compute Brewster's angle (angle of zero p-polarization reflection)."""
        return float(np.degrees(np.arctan(n2 / n1)))
