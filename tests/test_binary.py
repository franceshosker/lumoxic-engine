"""Tests for binary encoding and analysis."""

from lumoxic.binary.encoder import AngleEncoder
from lumoxic.binary.stream import StreamAssembler
from lumoxic.binary.patterns import PatternAnalyzer


def test_encoder_threshold():
    enc = AngleEncoder(threshold=45.0)
    assert enc.encode(50.0) == 1
    assert enc.encode(30.0) == 0
    assert enc.encode(45.0) == 1
    assert enc.encode(44.9) == 0


def test_encoder_batch():
    enc = AngleEncoder(threshold=45.0)
    result = enc.encode_batch([30, 50, 60, 20, 80])
    assert result == [0, 1, 1, 0, 1]


def test_encoder_to_string():
    enc = AngleEncoder(threshold=45.0)
    s = enc.encode_to_string([30, 50, 60, 20, 80])
    assert s == "01101"


def test_stream_assembler():
    sa = StreamAssembler()
    sa.extend([1, 0, 1, 1, 0, 0, 1, 0])
    assert sa.as_string == "10110010"
    assert sa.length == 8
    assert sa.to_hex() == "b2"


def test_pattern_entropy():
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    e = PatternAnalyzer.entropy(bits)
    assert abs(e - 1.0) < 0.01  # Maximum entropy for balanced bits


def test_pattern_frequency():
    bits = [1, 1, 1, 0, 0, 0]
    freq = PatternAnalyzer.frequency(bits)
    assert abs(freq[0] - 0.5) < 0.01
    assert abs(freq[1] - 0.5) < 0.01


def test_run_lengths():
    bits = [1, 1, 0, 0, 0, 1]
    runs = PatternAnalyzer.run_lengths(bits)
    assert runs == [(1, 2), (0, 3), (1, 1)]
