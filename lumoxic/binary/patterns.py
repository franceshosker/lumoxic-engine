"""Pattern recognition in binary sequences."""

from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


class PatternAnalyzer:
    """Analyze patterns in binary streams produced by the bounce engine."""

    @staticmethod
    def frequency(bits: List[int]) -> Dict[int, float]:
        """Compute bit frequency distribution."""
        if not bits:
            return {0: 0.0, 1: 0.0}
        c = Counter(bits)
        n = len(bits)
        return {0: c.get(0, 0) / n, 1: c.get(1, 0) / n}

    @staticmethod
    def entropy(bits: List[int]) -> float:
        """Compute Shannon entropy of the bit stream."""
        if not bits:
            return 0.0
        freq = PatternAnalyzer.frequency(bits)
        h = 0.0
        for p in freq.values():
            if p > 0:
                h -= p * np.log2(p)
        return h

    @staticmethod
    def autocorrelation(bits: List[int], max_lag: int = 10) -> List[float]:
        """Compute autocorrelation at different lags."""
        arr = np.array(bits, dtype=float)
        arr = arr - arr.mean()
        n = len(arr)
        if n < 2:
            return [1.0]
        var = np.var(arr)
        if var == 0:
            return [1.0] * min(max_lag, n)
        result = []
        for lag in range(min(max_lag, n)):
            if lag == 0:
                result.append(1.0)
            else:
                c = np.sum(arr[: n - lag] * arr[lag:]) / ((n - lag) * var)
                result.append(float(c))
        return result

    @staticmethod
    def run_lengths(bits: List[int]) -> List[Tuple[int, int]]:
        """Compute run-length encoding: list of (bit_value, run_length)."""
        if not bits:
            return []
        runs = []
        current = bits[0]
        length = 1
        for b in bits[1:]:
            if b == current:
                length += 1
            else:
                runs.append((current, length))
                current = b
                length = 1
        runs.append((current, length))
        return runs

    @staticmethod
    def find_repeating(bits: List[int], pattern_length: int = 4) -> Dict[str, int]:
        """Find repeating patterns of a given length."""
        if len(bits) < pattern_length:
            return {}
        patterns: Dict[str, int] = {}
        for i in range(len(bits) - pattern_length + 1):
            p = "".join(str(b) for b in bits[i : i + pattern_length])
            patterns[p] = patterns.get(p, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: -x[1]))
