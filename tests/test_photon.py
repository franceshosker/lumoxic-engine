"""Tests for photon emission and processing."""

from lumoxic.photon.emitter import PhotonEmitter
from lumoxic.optics.wavelength import WavelengthAnalyzer
from lumoxic.optics.refraction import RefractionCalculator


def test_emitter():
    emitter = PhotonEmitter(seed=42)
    photon = emitter.emit(wavelength=450, angle=37)
    assert photon.wavelength == 450
    assert photon.angle == 37


def test_emitter_batch():
    emitter = PhotonEmitter(seed=42)
    photons = emitter.emit_batch(10, wavelength=500)
    assert len(photons) == 10
    for p in photons:
        assert p.wavelength == 500
        assert 5 <= p.angle <= 85


def test_wavelength_color():
    assert WavelengthAnalyzer.color_name(450) == "blue"
    assert WavelengthAnalyzer.color_name(550) == "green"
    assert WavelengthAnalyzer.color_name(650) == "red"


def test_wavelength_energy():
    e = WavelengthAnalyzer.energy_ev(450)
    assert 2.5 < e < 3.0  # ~2.76 eV for 450nm


def test_snells_law():
    angle_r = RefractionCalculator.snells_law(30, 1.0, 1.5)
    assert angle_r is not None
    assert 15 < angle_r < 25


def test_critical_angle():
    ca = RefractionCalculator.critical_angle(1.5, 1.0)
    assert ca is not None
    assert 40 < ca < 45
