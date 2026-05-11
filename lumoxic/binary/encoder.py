"""Angle-to-bit encoding with configurable thresholds."""

from typing import List
import numpy as np

from lumoxic.core.constants import DEFAULT_THRESHOLD


class AngleEncoder:
    """Encode bounce angles into binary bits.

    The fundamental operation: angle >= threshold → 1, angle < threshold → 0.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def encode(self, angle: float) -> int:
        """Encode a single angle to a bit."""
        return 1 if angle >= self.threshold else 0

    def encode_batch(self, angles: List[float] | np.ndarray) -> List[int]:
        """Encode a list of angles to bits."""
        arr = np.asarray(angles)
        return (arr >= self.threshold).astype(int).tolist()

    def encode_to_string(self, angles: List[float] | np.ndarray) -> str:
        """Encode angles directly to a binary string."""
        return "".join(str(b) for b in self.encode_batch(angles))

    def encode_to_bytes(self, angles: List[float] | np.ndarray) -> bytes:
        """Encode angles to packed bytes (8 bits per byte)."""
        bits = self.encode_batch(angles)
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= bits[i + j] << (7 - j)
            result.append(byte)
        return bytes(result)

    def decode_byte(self, b: int) -> List[int]:
        """Decode a single byte back to 8 bits."""
        return [(b >> (7 - i)) & 1 for i in range(8)]
