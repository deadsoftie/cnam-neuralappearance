# SPDX-License-Identifier: Apache-2.0

"""GPU resources for the NTC SDK training forward pass (``ntc_forward.slang``).

Inputs are plain arrays: latent values per neural mip (FP16, shape
(features, height, width), in [0, 1]), one ``(weights, biases)`` pair per
layer (FP16, weights of shape (outputs, inputs)), the channel mapping
(scale, bias per channel, float32) and one ``(width, height, neural mip,
position scale, position lod)`` entry per color mip.

Buffer layouts: latents per neural mip, one plane per feature; weights per
layer row-major, then all bias vectors (``WEIGHT_OFFSETS``, ``BIAS_OFFSETS``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import slangpy as spy

MAX_COLOR_MIPS = 16
MAX_NEURAL_MIPS = 8
LAYER_SHAPES = ((48, 64), (64, 48), (48, 32), (32, 16))
WEIGHT_OFFSETS = (0, 3072, 6144, 7680)
BIAS_OFFSETS = (8192, 8256, 8304, 8336)
NUM_PARAMS = 8352
NUM_OUTPUTS = 16
STAGE_FP8 = 0
STAGE_INT8 = 1
MODULE = Path(__file__).with_name('ntc_forward.slang')

ColorMip = tuple[int, int, int, float, float]
Layer = tuple[np.ndarray, np.ndarray]


def load_module(device: spy.Device) -> spy.Module:
    return spy.Module.load_from_file(device, str(MODULE))


def check_inputs(latents: Sequence[np.ndarray], layers: Sequence[Layer], color_mips: Sequence[ColorMip]) -> int:
    """Validate shapes; return the feature count."""
    num_features = latents[0].shape[0]
    if any(lat.shape[0] != num_features for lat in latents) or not 0 < num_features <= 16:
        raise ValueError('every neural mip needs the same feature count, 1 to 16')
    if len(latents) > MAX_NEURAL_MIPS or len(color_mips) > MAX_COLOR_MIPS:
        raise ValueError(f'{len(latents)} neural mips, {len(color_mips)} color mips')
    if tuple((w.shape[1], w.shape[0]) for w, _ in layers) != LAYER_SHAPES:
        raise ValueError('unexpected network shape')
    if any(n + 1 >= len(latents) for _, _, n, _, _ in color_mips):
        raise ValueError('a color mip uses a neural mip pair the latents do not have')
    return num_features


def neural_mip_table(latents: Sequence[np.ndarray]) -> list[spy.int4]:
    """(width, height, first element) of each neural mip in the latent buffer."""
    table, start = [], 0
    for lat in latents:
        table.append(spy.int4(lat.shape[2], lat.shape[1], start, 0))
        start += lat.size
    return table


def flatten_latents(latents: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(lat, np.float16).astype(np.float32).ravel() for lat in latents])


def flatten_layers(layers: Sequence[Layer]) -> np.ndarray:
    parts = [np.asarray(w, np.float16).astype(np.float32).ravel() for w, _ in layers]
    parts += [np.asarray(b, np.float16).astype(np.float32).ravel() for _, b in layers]
    return np.concatenate(parts)


def unflatten_layers(values: np.ndarray) -> list[Layer]:
    layers = []
    for (n_in, n_out), w, b in zip(LAYER_SHAPES, WEIGHT_OFFSETS, BIAS_OFFSETS, strict=True):
        layers.append((values[w:w + n_in * n_out].reshape(n_out, n_in).astype(np.float16),
                       values[b:b + n_out].astype(np.float16)))
    return layers


def network_uniforms(
    latents: spy.Buffer,
    weights: spy.Buffer,
    color_mips: Sequence[ColorMip],
    neural_mips: Sequence[spy.int4],
    mapping: np.ndarray,
    num_features: int,
    stage: int,
    weight_start: int = 0,
) -> dict:
    """Fields of the Slang ``NtcNetwork`` struct. ``weight_start``: first element of the weight set used."""
    mips = [spy.int4(0, 0, 0, 0)] * MAX_COLOR_MIPS
    positions = [spy.float2(0, 0)] * MAX_COLOR_MIPS
    for level, (w, h, n, scale, lod) in enumerate(color_mips):
        mips[level] = spy.int4(w, h, n, 0)
        positions[level] = spy.float2(scale, lod)
    full = np.tile(np.array([1, 0], dtype=np.float32), (NUM_OUTPUTS, 1))
    full[: len(mapping)] = mapping
    return {
        'latents': latents,
        'weights': weights,
        'color_mips': mips,
        'positions': positions,
        'neural_mips': list(neural_mips) + [spy.int4(0, 0, 0, 0)] * (MAX_NEURAL_MIPS - len(neural_mips)),
        'weight_offsets': spy.int4(*(weight_start + o for o in WEIGHT_OFFSETS)),
        'bias_offsets': spy.int4(*(weight_start + o for o in BIAS_OFFSETS)),
        'mapping': [spy.float2(float(s), float(b)) for s, b in full],
        'num_features': num_features,
        'stage': stage,
        'one': 1.0,
    }


def create_buffer(device: spy.Device, data: np.ndarray) -> spy.Buffer:
    return device.create_buffer(
        usage=(spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source
               | spy.BufferUsage.copy_destination),
        data=np.ascontiguousarray(data),
    )


class NtcNetwork:
    def __init__(
        self,
        device: spy.Device,
        latents: Sequence[np.ndarray],
        layers: Sequence[Layer],
        mapping: np.ndarray,
        color_mips: Sequence[ColorMip],
        stage: int = STAGE_FP8,
    ):
        self.device = device
        self.num_features = check_inputs(latents, layers, color_mips)
        self.stage = stage
        self.color_mips = list(color_mips)
        self.mapping = np.asarray(mapping, np.float32)
        self.neural_mips = neural_mip_table(latents)
        self.latents = create_buffer(device, flatten_latents(latents))
        self.weights = create_buffer(device, flatten_layers(layers))

    def get_this(self) -> dict:
        return network_uniforms(self.latents, self.weights, self.color_mips, self.neural_mips, self.mapping,
                                self.num_features, self.stage)

    def forward_mip(self, module: spy.Module, mip: int) -> np.ndarray:
        """Linear-space latents of every texel of color mip ``mip``, shape (height, width, 8)."""
        w, h = self.color_mips[mip][:2]
        a = spy.Tensor.empty(self.device, (h, w), 'float4')
        b = spy.Tensor.empty(self.device, (h, w), 'float4')
        module.ntc_forward_mip(self.get_this(), mip, spy.grid((h, w)), a, b)
        return np.concatenate([a.to_numpy(), b.to_numpy()], axis=-1).reshape(h, w, 8)
