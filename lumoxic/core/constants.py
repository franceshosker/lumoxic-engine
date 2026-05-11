"""Physical constants and default configuration for photon computing."""

SPEED_OF_LIGHT = 299_792_458  # m/s
PLANCK_CONSTANT = 6.626e-34  # J·s
BOLTZMANN_CONSTANT = 1.381e-23  # J/K

DEFAULT_THRESHOLD = 45.0  # degrees — the binary decision boundary
DEFAULT_WAVELENGTH = 450.0  # nm — deep blue, peak efficiency
MIN_WAVELENGTH = 380.0  # nm — UV boundary
MAX_WAVELENGTH = 700.0  # nm — red boundary

SURFACE_MATERIALS = {
    "glass": {"reflection_coefficient": 0.92, "refractive_index": 1.52},
    "mirror": {"reflection_coefficient": 0.99, "refractive_index": 2.42},
    "crystal": {"reflection_coefficient": 0.95, "refractive_index": 1.77},
    "silicon": {"reflection_coefficient": 0.88, "refractive_index": 3.42},
    "diamond": {"reflection_coefficient": 0.97, "refractive_index": 2.42},
}
