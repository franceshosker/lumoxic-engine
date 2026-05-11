"""Surface definitions and material properties."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from lumoxic.core.types import Surface
from lumoxic.core.constants import SURFACE_MATERIALS


@dataclass
class SurfaceConfig:
    material: str = "glass"
    position: float = 0.0
    orientation: str = "horizontal"  # horizontal or vertical
    width: Optional[float] = None

    @property
    def reflection_coefficient(self) -> float:
        return SURFACE_MATERIALS.get(self.material, SURFACE_MATERIALS["glass"])[
            "reflection_coefficient"
        ]

    @property
    def refractive_index(self) -> float:
        return SURFACE_MATERIALS.get(self.material, SURFACE_MATERIALS["glass"])[
            "refractive_index"
        ]


def create_surface(config: SurfaceConfig) -> Surface:
    """Create a Surface from a SurfaceConfig."""
    if config.orientation == "vertical":
        normal = np.array([1.0, 0.0])
    else:
        normal = np.array([0.0, 1.0])

    return Surface(
        name=f"{config.material}_{config.orientation}_{config.position}",
        reflection_coefficient=config.reflection_coefficient,
        position=config.position,
        normal=normal,
        material=config.material,
    )


def create_chamber(
    width: float = 20.0,
    height: float = 10.0,
    material: str = "glass",
) -> list[Surface]:
    """Create a rectangular bounce chamber with the given dimensions."""
    return [
        Surface("bottom", SURFACE_MATERIALS[material]["reflection_coefficient"], 0.0, np.array([0.0, 1.0]), material),
        Surface("top", SURFACE_MATERIALS[material]["reflection_coefficient"], height, np.array([0.0, -1.0]), material),
        Surface("left", SURFACE_MATERIALS[material]["reflection_coefficient"], 0.0, np.array([1.0, 0.0]), material),
        Surface("right", SURFACE_MATERIALS[material]["reflection_coefficient"], width, np.array([-1.0, 0.0]), material),
    ]
