# SPDX-License-Identifier: Apache-2.0

"""Random number generation of the NTC SDK compression (``CudaUtils.h``, ``CudaRandomGen.cu``,
``TextureSet.cpp``).

``Mt19937`` is ``std::mt19937`` seeded with one integer; with MSVC,
``std::uniform_int_distribution<uint32_t>`` over the full range returns its raw outputs.
``hash_*`` functions are ``HashBasedRNG`` for arrays of generators.
"""

from __future__ import annotations

import numpy as np

U32 = np.uint32


class Mt19937:
    def __init__(self, seed: int):
        mt = [0] * 624
        mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mt = mt
        self.index = 624

    def _twist(self) -> None:
        mt = self.mt
        for i in range(624):
            y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.index = 0

    def next_uint32(self) -> int:
        if self.index >= 624:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def _jenkins(a: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        a = (a + U32(0x7ED55D16)) + (a << U32(12))
        a = (a ^ U32(0xC761C23C)) ^ (a >> U32(19))
        a = (a + U32(0x165667B1)) + (a << U32(5))
        a = (a + U32(0xD3A2646C)) ^ (a << U32(9))
        a = (a + U32(0xFD7046C5)) + (a << U32(3))
        a = (a ^ U32(0xB55A4F09)) ^ (a >> U32(16))
    return a


def _rot(x: np.ndarray, r: int) -> np.ndarray:
    return (x << U32(r)) | (x >> U32(32 - r))


def hash_seed(linear_index: np.ndarray, offset: np.ndarray | int) -> np.ndarray:
    """``m_seed`` of ``HashBasedRNG(linearIndex, offset)``."""
    with np.errstate(over='ignore'):
        return _jenkins(np.asarray(linear_index, U32)) + np.asarray(offset, U32)


def hash_uint(seed: np.ndarray, index: int) -> np.ndarray:
    """``Murmur3Hash`` for the generator state ``(m_seed, m_index)``."""
    with np.errstate(over='ignore'):
        k = U32(index) * U32(0xCC9E2D51)
        k = _rot(np.asarray(k, U32), 15) * U32(0x1B873593)
        h = np.asarray(seed, U32) ^ k
        h = _rot(h, 13) * U32(5) + U32(0xE6546B64)
        h ^= U32(4)
        h ^= h >> U32(16)
        h *= U32(0x85EBCA6B)
        h ^= h >> U32(13)
        h *= U32(0xC2B2AE35)
        h ^= h >> U32(16)
    return h


def hash_float(seed: np.ndarray, index: int) -> np.ndarray:
    """``NextFloat`` at ``m_index == index``: 23 random mantissa bits, value in [0, 1)."""
    bits = (hash_uint(seed, index) & U32((1 << 23) - 1)) | U32(0x3F800000)
    return bits.view(np.float32) - np.float32(1)


def fill_normal(length: int, seed: int, scale: float, bias: float, lo: float, hi: float) -> np.ndarray:
    """``FillRandomNormalHalf``: Box-Muller pairs per thread, clamped, as FP16 values.

    ``logf``, ``cosf`` and ``sinf`` are evaluated in float32 by numpy; they can differ from CUDA in
    the last bit, so a value can land on the neighboring FP16 value.
    """
    pairs = (length + 1) // 2
    s = hash_seed(np.arange(pairs, dtype=U32), seed)
    u = hash_float(s, 1)
    v = hash_float(s, 2)
    f = np.float32
    r = np.sqrt(np.maximum(f(0), f(-2) * np.log(np.maximum(u, f(1e-20)))))
    theta = f(2) * f(np.pi) * v
    out = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(f).ravel()[:length]
    out = np.minimum(np.maximum(out * f(scale) + f(bias), f(lo)), f(hi))
    out[~np.isfinite(out)] = 0
    return out.astype(np.float16)
