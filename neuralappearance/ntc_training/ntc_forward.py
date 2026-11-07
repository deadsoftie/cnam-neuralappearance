# SPDX-License-Identifier: Apache-2.0

"""GPU resources for the NTC SDK training forward pass (``ntc_forward.slang``).

Inputs are plain arrays: latent values per neural mip (FP16, shape
(features, height, width), in [0, 1]), one ``(weights, biases)`` pair per
layer (FP16, weights of shape (outputs, inputs)), the channel mapping
(scale, bias per channel, float32) and one ``(width, height, neural mip,
position scale, position lod)`` entry per color mip.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import slangpy as spy

MAX_COLOR_MIPS = 16
MAX_NEURAL_MIPS = 8
LAYER_SHAPES = ((48, 64), (64, 48), (48, 32), (32, 16))
NUM_OUTPUTS = 16
STAGE_FP8 = 0
STAGE_INT8 = 1
MODULE = Path(__file__).with_name('ntc_forward.slang')

ColorMip = tuple[int, int, int, float, float]


def load_module(device: spy.Device) -> spy.Module:
    return spy.Module.load_from_file(device, str(MODULE))


class NtcNetwork:
    type_name = 'NtcNetwork'

    def __init__(
        self,
        device: spy.Device,
        latents: Sequence[np.ndarray],
        layers: Sequence[tuple[np.ndarray, np.ndarray]],
        mapping: np.ndarray,
        color_mips: Sequence[ColorMip],
        stage: int = STAGE_FP8,
    ):
        num_features = latents[0].shape[0]
        if any(lat.shape[0] != num_features for lat in latents) or not 0 < num_features <= 16:
            raise ValueError('every neural mip needs the same feature count, 1 to 16')
        if len(latents) > MAX_NEURAL_MIPS or len(color_mips) > MAX_COLOR_MIPS:
            raise ValueError(f'{len(latents)} neural mips, {len(color_mips)} color mips')
        if tuple((w.shape[1], w.shape[0]) for w, _ in layers) != LAYER_SHAPES:
            raise ValueError('unexpected network shape')
        if any(n + 1 >= len(latents) for _, _, n, _, _ in color_mips):
            raise ValueError('a color mip uses a neural mip pair the latents do not have')
        self.device = device
        self.num_features = num_features
        self.stage = stage
        self.color_mips = list(color_mips)

        offsets, start = [], 0
        for lat in latents:
            offsets.append(start)
            start += lat.size
        self.neural_mips = [spy.int4(lat.shape[2], lat.shape[1], off, 0) for lat, off in zip(latents, offsets)]
        self.latents = device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=np.concatenate([np.asarray(lat, np.float16).astype(np.float32).ravel() for lat in latents]),
        )

        parts = [np.asarray(w, np.float16).astype(np.float32).ravel() for w, _ in layers]
        parts += [np.asarray(b, np.float16).astype(np.float32).ravel() for _, b in layers]
        bounds = np.cumsum([0] + [p.size for p in parts])
        self.weight_offsets = spy.int4(*(int(v) for v in bounds[:4]))
        self.bias_offsets = spy.int4(*(int(v) for v in bounds[4:8]))
        self.weights = device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=np.concatenate(parts),
        )

        full = np.tile(np.array([1, 0], dtype=np.float32), (NUM_OUTPUTS, 1))
        full[: len(mapping)] = mapping
        self.mapping = [spy.float2(float(s), float(b)) for s, b in full]

    def get_this(self) -> dict:
        mips = [spy.int4(0, 0, 0, 0)] * MAX_COLOR_MIPS
        positions = [spy.float2(0, 0)] * MAX_COLOR_MIPS
        for level, (w, h, n, scale, lod) in enumerate(self.color_mips):
            mips[level] = spy.int4(w, h, n, 0)
            positions[level] = spy.float2(scale, lod)
        neural = self.neural_mips + [spy.int4(0, 0, 0, 0)] * (MAX_NEURAL_MIPS - len(self.neural_mips))
        return {
            'latents': self.latents,
            'weights': self.weights,
            'color_mips': mips,
            'positions': positions,
            'neural_mips': neural,
            'weight_offsets': self.weight_offsets,
            'bias_offsets': self.bias_offsets,
            'mapping': self.mapping,
            'num_features': self.num_features,
            'stage': self.stage,
            'one': 1.0,
        }

    def forward_mip(self, module: spy.Module, mip: int) -> np.ndarray:
        """Linear-space latents of every texel of color mip ``mip``, shape (height, width, 8)."""
        w, h = self.color_mips[mip][:2]
        a = spy.Tensor.empty(self.device, (h, w), 'float4')
        b = spy.Tensor.empty(self.device, (h, w), 'float4')
        module.ntc_forward_mip(self.get_this(), mip, spy.grid((h, w)), a, b)
        return np.concatenate([a.to_numpy(), b.to_numpy()], axis=-1).reshape(h, w, 8)
