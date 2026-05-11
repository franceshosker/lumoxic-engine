"""Tests for the unified Client API."""

from lumoxic import Client


def test_client_process():
    client = Client(seed=42)
    result = client.process(wavelength=450, angle=37, bounces=8)
    assert result.binary_stream.length > 0
    assert result.photon_count == 1


def test_client_simulate():
    client = Client(seed=42)
    result = client.simulate(photon_count=10, max_bounces=4)
    assert result.photon_count == 10
    assert result.binary_stream.length > 0


def test_client_encode():
    client = Client()
    binary = client.encode([30.0, 50.0, 60.0, 20.0, 80.0])
    assert binary == "01100" or binary == "01101"  # depends on threshold


def test_client_analyze():
    client = Client()
    analysis = client.analyze([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    assert "entropy" in analysis
    assert "frequency" in analysis
    assert 0 < analysis["entropy"] <= 1.0
