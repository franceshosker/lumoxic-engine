"""Tests for the Binary Bounce Engine."""

from lumoxic.bounce.engine import BinaryBounceEngine
from lumoxic.core.types import Photon


def test_engine_fire_photon():
    engine = BinaryBounceEngine(seed=42)
    photon = Photon(wavelength=450, angle=37)
    result = engine.fire_photon(photon, max_bounces=5)
    assert len(result.bounces) > 0
    assert result.binary_stream.length == len(result.bounces)


def test_engine_simulate():
    engine = BinaryBounceEngine(seed=42)
    result = engine.simulate(photon_count=10, max_bounces=3)
    assert result.photon_count == 10
    assert result.binary_stream.length > 0


def test_threshold_boundary():
    engine = BinaryBounceEngine(threshold=45.0, seed=42)
    result = engine.simulate(photon_count=100)
    bits = result.binary_stream.bits
    assert 0 in bits
    assert 1 in bits
