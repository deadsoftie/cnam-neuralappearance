# SPDX-License-Identifier: Apache-2.0

"""NTC grid and decode network training, step for step as ``TextureSet::RunCompressionSteps``.

Differences from the SDK: float32 backward, latent gradients summed in 64-bit fixed
point, latent Adam bias correction with step + 1 (as the network's). The loss is the
SDK's L2 against a reference pyramid; its gradient reaches the network output without
the channel mapping scale, as in the SDK.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from ntc_training import sdk_rng
from ntc_training.ntc_forward import (
    LAYER_SHAPES,
    NUM_PARAMS,
    STAGE_FP8,
    STAGE_INT8,
    ColorMip,
    Layer,
    check_inputs,
    create_buffer,
    flatten_latents,
    flatten_layers,
    network_uniforms,
    neural_mip_table,
    unflatten_layers,
)

MODULE = Path(__file__).with_name('ntc_train.slang')
F32 = np.float32
NUM_COLUMNS = sum(o for _, o in LAYER_SHAPES)
FREEZE_RATIO = F32(0.95)
FP8_RATIO = F32(0.96)
INT8_RATIO = F32(0.98)


@dataclass(frozen=True)
class Settings:
    seed: int
    steps: int = 100000
    steps_per_iteration: int = 1000
    network_learning_rate: float = 0.005
    grid_learning_rate: float = 0.1
    batch: int = 64 * 1024
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    chunk: int = 256
    valid_channels: int = 8


@dataclass
class Iteration:
    step: int
    loss: float
    loss_scale: float
    milliseconds_per_step: float


def cosine(step: int, lr_max: float, steps: int) -> F32:
    return F32(0.5) * F32(lr_max) * (F32(1) + np.cos(F32(step) * F32(np.pi) / F32(steps)))


def adam_constants(s: Settings, loss_scale: float, step: float, lr: F32, seed: int, half_offset: int) -> dict:
    bc1 = F32(1) - F32(s.beta1) ** F32(step)
    bc2 = F32(1) - F32(s.beta2) ** F32(step)
    return {
        'inv_loss_scale': float(F32(1) / F32(loss_scale)),
        'beta1': s.beta1,
        'beta2': s.beta2,
        'epsilon': s.epsilon,
        'step_size': float(lr * (F32(1) / bc1)),
        'inv_sqrt_bias_correction2': float(F32(1) / np.sqrt(bc2)),
        'seed': seed,
        'half_offset': half_offset,
    }


def mip_cdf(color_mips: Sequence[ColorMip]) -> list[float]:
    """``mipInfos[].cdf``: pdf proportional to ``max(texels, 512)``, float32 as the SDK."""
    pdf = [F32(max(w * h, 512)) for w, h, *_ in color_mips]
    total = F32(0)
    for p in pdf:
        total = F32(total + p)
    cdf, prev = [], F32(0)
    for p in pdf:
        prev = F32(p / total + prev)
        cdf.append(float(prev))
    return cdf + [0.0] * (16 - len(cdf))


def mma_offsets(rows: int, cols: int) -> np.ndarray:
    """Half index in the SDK's MMA weight layout (``tin::host::HMatrixB``) of each (col, row)."""
    r = np.arange(rows)[None, :]
    c = np.arange(cols)[:, None]
    rp = r // 2
    mma_idx = (c // 16) * (rows // 16) + rp // 8
    sub_r, sub_c = rp % 8, c % 16
    return ((32 * mma_idx + (sub_c % 8) * 4 + sub_r % 4) * 4 + (sub_c // 8) * 2 + sub_r // 4) * 2 + r % 2


def sdk_initial_state(grid_shapes: Sequence[tuple[int, int]], num_features: int,
                      seed: int) -> tuple[list[np.ndarray], list[Layer]]:
    """SDK random initialization (``BeginCompression``) for ``(width, height)`` neural mips."""
    pixels = [w * h for w, h in grid_shapes]
    total = sum(pixels)
    flat = sdk_rng.fill_normal(total * num_features, seed, 0.25, 0.5, 0.0, 1.0)
    latents, start = [], 0
    for (w, h), n in zip(grid_shapes, pixels, strict=True):
        planes = [flat[(f // 2) * total * 2 + (start + np.arange(n)) * 2 + f % 2].reshape(h, w)
                  for f in range(num_features)]
        latents.append(np.stack(planes))
        start += n
    layers = []
    for index, (n_in, n_out) in enumerate(LAYER_SHAPES):
        values = sdk_rng.fill_normal(n_in * n_out, seed + 1 + index, float(np.sqrt(F32(2) / F32(n_in))), 0.0,
                                     -1000.0, 1000.0)
        layers.append((values[mma_offsets(n_in, n_out)], np.zeros(n_out, np.float16)))
    return latents, layers


class Trainer:
    def __init__(
        self,
        device: spy.Device,
        latents: Sequence[np.ndarray],
        layers: Sequence[Layer],
        mapping: np.ndarray,
        color_mips: Sequence[ColorMip],
        reference: Sequence[np.ndarray],
        settings: Settings,
    ):
        """``reference``: per color mip, shape (height, width, 8), the dense latent rounded to FP16."""
        if settings.batch % settings.chunk or settings.batch % 32:
            raise ValueError('batch must be a multiple of the chunk and of 32')
        self.device = device
        self.module = spy.Module.load_from_file(device, str(MODULE))
        self.s = settings
        self.num_features = check_inputs(latents, layers, color_mips)
        self.color_mips = list(color_mips)
        self.mapping = np.asarray(mapping, F32)
        self.neural_mips = neural_mip_table(latents)
        self.grid_shapes = [(lat.shape[2], lat.shape[1]) for lat in latents]
        self.total_pixels = sum(w * h for w, h in self.grid_shapes)
        if len(reference) != len(color_mips):
            raise ValueError('one reference image per color mip')
        for (w, h, *_), ref in zip(color_mips, reference, strict=True):
            if ref.shape != (h, w, 8):
                raise ValueError(f'reference of shape {ref.shape}, expected {(h, w, 8)}')

        lat = flatten_latents(latents)
        weights = np.tile(flatten_layers(layers), 2)
        b = settings.batch
        self.buffers = {
            'latents_base': create_buffer(device, lat),
            'latents_q': create_buffer(device, lat),
            'latent_m1': create_buffer(device, np.zeros_like(lat)),
            'latent_m2': create_buffer(device, np.zeros_like(lat)),
            'latent_grad': create_buffer(device, np.zeros(lat.size, np.int64)),
            'grad_mask': create_buffer(device, np.zeros((self.total_pixels + 31) // 32, np.uint32)),
            'weights_base': create_buffer(device, weights),
            'weights_q': create_buffer(device, weights),
            'weight_m1': create_buffer(device, np.zeros(NUM_PARAMS, F32)),
            'weight_m2': create_buffer(device, np.zeros(NUM_PARAMS, F32)),
            'weight_grad': create_buffer(device, np.zeros(NUM_PARAMS, F32)),
            'texels': create_buffer(device, np.zeros((b, 4), np.int32)),
            'acts': create_buffer(device, np.zeros(b * 192, F32)),
            'grads': create_buffer(device, np.zeros(b * 160, F32)),
            'texel_loss': create_buffer(device, np.zeros(b, F32)),
            'partials': create_buffer(device, np.zeros(b // settings.chunk * NUM_PARAMS, F32)),
            'reference': create_buffer(device, np.concatenate(
                [np.asarray(r, np.float16).astype(F32).ravel() for r in reference])),
        }
        offsets = np.cumsum([0] + [w * h * 8 for w, h, *_ in color_mips])[:-1]
        pixel_offsets = np.cumsum([0] + [w * h for w, h in self.grid_shapes])[:-1]
        self.state_fields = {
            **self.buffers,
            'reference_offsets': [int(v) for v in offsets] + [0] * (16 - len(offsets)),
            'pixel_offsets': [int(v) for v in pixel_offsets] + [0] * (8 - len(pixel_offsets)),
            'num_neural_mips': len(self.grid_shapes),
            'total_pixels': self.total_pixels,
            'batch': b,
            'chunk': settings.chunk,
        }
        self.sampling = {
            'texels': self.buffers['texels'],
            'cdf': mip_cdf(color_mips),
            'num_mips': len(color_mips),
            'width': color_mips[0][0],
            'height': color_mips[0][1],
            'seed': 0,
        }
        self.rng = sdk_rng.Mt19937(settings.seed)
        self.step = 0
        self.loss_scale = 256.0
        self.history: list[Iteration] = []

    def network(self, stage: int, weight_start: int) -> dict:
        return network_uniforms(self.buffers['latents_q'], self.buffers['weights_q'], self.color_mips,
                                self.neural_mips, self.mapping, self.num_features, stage, weight_start)

    def schedule(self) -> tuple[int, int, int]:
        steps = F32(self.s.steps)
        return int(steps * FREEZE_RATIO), int(steps * FP8_RATIO), int(steps * INT8_RATIO)

    def run_step(self) -> float | None:
        """One training step. Returns the loss when it is the last step of an iteration."""
        s, m, step = self.s, self.module, self.step
        freeze_step, fp8_step, int8_step = self.schedule()
        quantize_fp8 = fp8_step < step <= int8_step
        quantize_int8 = step > int8_step
        half = NUM_PARAMS if quantize_int8 else 0

        if step == fp8_step:
            for name in ('weights_base', 'weights_q'):
                values = self.buffers[name].to_numpy().view(F32)
                values[NUM_PARAMS:] = values[:NUM_PARAMS]
                self.buffers[name].copy_from_numpy(values)
        if quantize_fp8 or quantize_int8:
            m.ntc_fake_quantize(self.state_fields, half, int(quantize_fp8), spy.grid((NUM_COLUMNS,)))

        m.ntc_clear_mask(self.state_fields, spy.grid(((self.total_pixels + 31) // 32,)))
        self.sampling['seed'] = self.rng.next_uint32()
        m.ntc_sample(self.sampling, spy.grid((s.batch,)))

        stage = STAGE_INT8 if quantize_int8 else STAGE_FP8
        loss_params = {'gradient_scale': float(F32(self.loss_scale) / F32(s.batch)),
                       'accumulate_latents': int(step < freeze_step)}
        m.ntc_train_texel(self.network(stage, half), self.state_fields, loss_params, spy.grid((s.batch,)))
        m.ntc_weight_chunk(self.state_fields, spy.grid((s.batch // s.chunk, NUM_PARAMS)))
        m.ntc_weight_sum(self.state_fields, spy.grid((NUM_PARAMS,)))

        network_lr = cosine(step, s.network_learning_rate, s.steps)
        m.ntc_network_adam(self.state_fields,
                           adam_constants(s, self.loss_scale, step + 1, network_lr, self.rng.next_uint32(), half),
                           spy.grid((NUM_PARAMS,)))
        if step == freeze_step:
            m.ntc_freeze(self.state_fields, spy.grid((self.total_pixels * self.num_features,)))
        elif step < freeze_step:
            grid_lr = cosine(step, s.grid_learning_rate, s.steps)
            m.ntc_latent_adam(self.network(stage, half), self.state_fields,
                              adam_constants(s, self.loss_scale, step + 1, grid_lr, self.rng.next_uint32(), 0),
                              spy.grid((self.num_features // 2, self.total_pixels)))

        self.device.wait()
        self.step += 1
        if self.step % s.steps_per_iteration and self.step != s.steps:
            return None
        loss = float(self.buffers['texel_loss'].to_numpy().view(F32).astype(np.float64).sum()) / s.batch
        return float(F32(loss) / F32(s.valid_channels))

    def run(self, on_iteration: Callable[[Iteration], None] | None = None) -> list[Iteration]:
        start = time.perf_counter()
        while self.step < self.s.steps:
            loss = self.run_step()
            if loss is None:
                continue
            self.device.wait()
            elapsed = time.perf_counter() - start
            item = Iteration(self.step, loss, self.loss_scale, elapsed * 1000 / self.s.steps_per_iteration)
            self.loss_scale = float(min(F32(32768), F32(128) / np.sqrt(F32(loss)))) if loss > 0 else 32768.0
            self.history.append(item)
            if on_iteration:
                on_iteration(item)
            start = time.perf_counter()
        return self.history

    def latents(self) -> list[np.ndarray]:
        """Quantized latents per neural mip, FP16 of shape (features, height, width)."""
        values = self.buffers['latents_q'].to_numpy().view(F32)
        out, start = [], 0
        for w, h in self.grid_shapes:
            n = self.num_features * w * h
            out.append(values[start:start + n].reshape(self.num_features, h, w).astype(np.float16))
            start += n
        return out

    def weight_sets(self) -> tuple[list[Layer], list[Layer]]:
        """FP16 weights the SDK quantizes at the end: the FP8 half, then the Int8 half."""
        values = self.buffers['weights_q'].to_numpy().view(F32)
        return unflatten_layers(values[:NUM_PARAMS]), unflatten_layers(values[NUM_PARAMS:])
