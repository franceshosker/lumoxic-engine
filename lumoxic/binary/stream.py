"""Binary stream assembly and manipulation."""

from typing import List
import numpy as np


class StreamAssembler:
    """Assemble and manipulate binary streams from encoded angles."""

    def __init__(self):
        self._buffer: List[int] = []

    def append(self, bit: int) -> None:
        self._buffer.append(bit & 1)

    def extend(self, bits: List[int]) -> None:
        self._buffer.extend(b & 1 for b in bits)

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def bits(self) -> List[int]:
        return self._buffer.copy()

    @property
    def as_string(self) -> str:
        return "".join(str(b) for b in self._buffer)

    @property
    def length(self) -> int:
        return len(self._buffer)

    def to_int(self) -> int:
        """Convert the binary stream to an integer."""
        result = 0
        for bit in self._buffer:
            result = (result << 1) | bit
        return result

    def to_hex(self) -> str:
        """Convert the binary stream to hexadecimal."""
        n = self.to_int()
        hex_len = (self.length + 3) // 4
        return f"{n:0{hex_len}x}"

    def chunk(self, size: int = 8) -> List[List[int]]:
        """Split the stream into chunks of the given size."""
        return [self._buffer[i : i + size] for i in range(0, len(self._buffer), size)]

    def hamming_distance(self, other: "StreamAssembler") -> int:
        """Compute Hamming distance between two streams."""
        min_len = min(self.length, other.length)
        return sum(
            a != b for a, b in zip(self._buffer[:min_len], other._buffer[:min_len])
        )
