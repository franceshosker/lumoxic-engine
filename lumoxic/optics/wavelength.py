"""Wavelength analysis and spectral decomposition."""

import numpy as np
from lumoxic.core.constants import MIN_WAVELENGTH, MAX_WAVELENGTH, SPEED_OF_LIGHT, PLANCK_CONSTANT


class WavelengthAnalyzer:
    """Analyze photon wavelengths and their properties."""

    @staticmethod
    def frequency(wavelength_nm: float) -> float:
        """Convert wavelength (nm) to frequency (Hz)."""
        return SPEED_OF_LIGHT / (wavelength_nm * 1e-9)

    @staticmethod
    def energy(wavelength_nm: float) -> float:
        """Compute photon energy (Joules) from wavelength."""
        return PLANCK_CONSTANT * WavelengthAnalyzer.frequency(wavelength_nm)

    @staticmethod
    def energy_ev(wavelength_nm: float) -> float:
        """Compute photon energy in electron-volts."""
        return WavelengthAnalyzer.energy(wavelength_nm) / 1.602e-19

    @staticmethod
    def color_name(wavelength_nm: float) -> str:
        """Get the visible color name for a wavelength."""
        if wavelength_nm < 380:
            return "ultraviolet"
        elif wavelength_nm < 450:
            return "violet"
        elif wavelength_nm < 495:
            return "blue"
        elif wavelength_nm < 570:
            return "green"
        elif wavelength_nm < 590:
            return "yellow"
        elif wavelength_nm < 620:
            return "orange"
        elif wavelength_nm < 700:
            return "red"
        return "infrared"

    @staticmethod
    def binary_yield(wavelength_nm: float) -> float:
        """Estimate binary conversion yield for a wavelength.

        Shorter wavelengths (blue/UV) produce higher yields due to
        greater photon energy and tighter bounce angles.
        """
        normalized = (wavelength_nm - MIN_WAVELENGTH) / (MAX_WAVELENGTH - MIN_WAVELENGTH)
        return float(np.clip(1.0 - 0.5 * normalized, 0.3, 1.0))

    @staticmethod
    def spectral_analysis(wavelengths: list[float]) -> dict:
        """Analyze a set of wavelengths."""
        arr = np.array(wavelengths)
        return {
            "count": len(arr),
            "mean_nm": float(arr.mean()),
            "std_nm": float(arr.std()),
            "min_nm": float(arr.min()),
            "max_nm": float(arr.max()),
            "mean_energy_ev": float(np.mean([WavelengthAnalyzer.energy_ev(w) for w in wavelengths])),
            "mean_yield": float(np.mean([WavelengthAnalyzer.binary_yield(w) for w in wavelengths])),
        }
