# SPDX-License-Identifier: Apache-2.0

"""NTC grid and decode network training, step for step as ``TextureSet::RunCompressionSteps``.

Differences from the SDK: matrix products on cooperative vectors (FP16 as the SDK's tensor cores,
rounding inside the hardware), weight gradients reduced per warp, then in a fixed order, latent
gradients summed in 64-bit fixed point, latent Adam bias correction with step + 1 (as the
network's). Steps are recorded as RTNAM records them: a block of steps per command buffer with a
global barrier after each kernel, several blocks queued, the device waited for only at the end of an
iteration, where the loss read back asynchronously sets the next loss scale. The loss is the SDK's
L2 against a reference pyramid; its gradient reaches the network output without the channel mapping
scale, as in the SDK. With a ``BsdfLoss`` the loss is the E2E BSDF loss instead, on the same texels;
its gradient reaches the outputs the same way, minus the gradient at the reference pyramid when one
is given (control variate). ``micro_batches`` above 1 averages the gradients of that many batches,
drawn with the same weights, into one update (not in the SDK).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from ntc_training import sdk_rng
from ntc_training.ntc_forward import (
    BIAS_OFFSETS,
    LAYER_SHAPES,
    NUM_PARAMS,
    STAGE_FP8,
    STAGE_INT8,
    WEIGHT_OFFSETS,
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
NUM_WEIGHTS = BIAS_OFFSETS[0]
COOP_ALIGNMENT = 64
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
    block_steps: int = 16
    blocks_in_flight: int = 8
    train_latents: bool = True
    micro_batches: int = 1


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


def aligned(n: int) -> int:
    return (n + COOP_ALIGNMENT - 1) // COOP_ALIGNMENT * COOP_ALIGNMENT


class CoopLayout:
    """FP16 training-optimal matrices of the four layers, then the bias vectors. ``index[p]``: FP16
    index of parameter ``p`` (``weights_q`` order) in a gradient slot."""

    def __init__(self, device: spy.Device):
        layout, half = spy.CoopVecMatrixLayout.training_optimal, spy.DataType.float16
        self.matrices, self.weight_offsets, start = [], [], 0
        for n_in, n_out in LAYER_SHAPES:
            desc = device.create_coop_vec_matrix_desc(n_out, n_in, layout, half, start)
            self.matrices.append(desc)
            self.weight_offsets.append(start)
            start = aligned(start + desc.size)
        self.weights_bytes = start
        self.bias_offsets = [2 * (o - BIAS_OFFSETS[0]) for o in BIAS_OFFSETS]
        self.biases_bytes = 2 * (NUM_PARAMS - NUM_WEIGHTS)
        self.slot_bytes = aligned(self.weights_bytes + self.biases_bytes)
        self.index = np.empty(NUM_PARAMS, np.int32)
        for l, (n_in, n_out) in enumerate(LAYER_SHAPES):
            rows = self.to_layout(device, l, np.repeat(np.arange(1, n_out + 1)[:, None], n_in, 1))
            cols = self.to_layout(device, l, np.repeat(np.arange(1, n_in + 1)[None, :], n_out, 0))
            at = np.nonzero(rows)[0]
            flat = (rows[at] - 1) * n_in + cols[at] - 1
            if not np.array_equal(np.sort(flat), np.arange(n_in * n_out)):
                raise RuntimeError(f'layer {l}: training-optimal layout does not hold every weight once')
            self.index[WEIGHT_OFFSETS[l] + flat] = self.weight_offsets[l] // 2 + at
            biases = np.arange(n_out)
            self.index[BIAS_OFFSETS[l] + biases] = (self.weights_bytes + self.bias_offsets[l]) // 2 + biases

    def source(self, layer: int, start: int) -> spy.CoopVecMatrixDesc:
        """Row-major FP32 matrix of ``layer`` in a parameter buffer whose weight set begins at ``start``."""
        n_in, n_out = LAYER_SHAPES[layer]
        return spy.CoopVecMatrixDesc({'rows': n_out, 'cols': n_in, 'layout': spy.CoopVecMatrixLayout.row_major,
                                      'element_type': spy.DataType.float32, 'size': n_in * n_out * 4,
                                      'offset': (start + WEIGHT_OFFSETS[layer]) * 4, 'row_col_stride': n_in * 4})

    def to_layout(self, device: spy.Device, layer: int, values: np.ndarray) -> np.ndarray:
        """Small integers of a row-major matrix, read back in the layer's training-optimal order."""
        src = device.create_buffer(usage=spy.BufferUsage.shader_resource,
                                   data=np.ascontiguousarray(values, F32).ravel())
        dst_desc = self.matrices[layer]
        dst = device.create_buffer(size=dst_desc.offset + dst_desc.size,
                                   usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
        encoder = device.create_command_encoder()
        encoder.clear_buffer(dst)
        encoder.convert_coop_vec_matrix(dst, dst_desc, src, self.source(layer, -WEIGHT_OFFSETS[layer]))
        device.submit_command_buffer(encoder.finish())
        device.wait()
        out = dst.to_numpy().view(np.float16)[dst_desc.offset // 2:]
        return np.rint(out.astype(F32)).astype(np.int64)


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
        reference: Sequence[np.ndarray] | None,
        settings: Settings,
        bsdf: Any = None,
    ):
        """``reference``: per color mip, shape (height, width, 8), the dense latent. The L2 loss uses it
        rounded to FP16; a ``bsdf`` (E2E loss) uses it as given as control variate, or runs without one when
        it is None."""
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
        if bsdf is None and reference is None:
            raise ValueError('the L2 loss needs a reference')
        self.bsdf = bsdf
        self.control_variate = bsdf is not None and reference is not None
        self.encoder = None
        if reference is not None:
            if len(reference) != len(color_mips):
                raise ValueError('one reference image per color mip')
            for (w, h, *_), ref in zip(color_mips, reference, strict=True):
                if ref.shape != (h, w, 8):
                    raise ValueError(f'reference of shape {ref.shape}, expected {(h, w, 8)}')
            dtype = np.float16 if bsdf is None else F32
            reference_values = np.concatenate([np.asarray(r, dtype).astype(F32).ravel() for r in reference])
        else:
            reference_values = np.zeros(1, F32)

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
            'output_grads': create_buffer(device, np.zeros(b * 16, F32)),
            'texel_loss': create_buffer(device, np.zeros(b, F32)),
            'partials': create_buffer(device, np.zeros(b // settings.chunk * NUM_PARAMS, F32)),
            'reference': create_buffer(device, reference_values),
        }
        self.coop = CoopLayout(device)
        coop_usage = (spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
                      | spy.BufferUsage.copy_destination)
        self.buffers |= {
            'coop_weights': device.create_buffer(size=self.coop.weights_bytes, usage=coop_usage),
            'coop_biases': device.create_buffer(size=self.coop.biases_bytes, usage=coop_usage),
            'grad_slots': device.create_buffer(size=b // 32 * self.coop.slot_bytes, usage=coop_usage),
            'slot_index': create_buffer(device, self.coop.index),
        }
        offsets = np.cumsum([0] + [w * h * 8 for w, h, *_ in color_mips])[:-1]
        pixel_offsets = np.cumsum([0] + [w * h for w, h in self.grid_shapes])[:-1]
        self.state_fields = {
            **self.buffers,
            'coop_weight_offsets': spy.int4(*self.coop.weight_offsets),
            'coop_bias_offsets': spy.int4(*self.coop.bias_offsets),
            'slot_bytes': self.coop.slot_bytes,
            'slot_bias_start': self.coop.weights_bytes,
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
        self.loss_readback = device.create_buffer(size=b * 4, usage=spy.BufferUsage.copy_destination,
                                                  memory_type=spy.MemoryType.read_back)
        self.fence = device.create_fence(0)
        self.fence_value = 0
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

    def dispatch(self, kernel, *args, **kwargs) -> None:
        """While steps are recorded, appends to their command buffer, else runs and waits."""
        if self.encoder is None:
            kernel(*args, **kwargs)
            self.device.wait()
            return
        kernel.append_to(self.encoder, *args, **kwargs)
        self.encoder.global_barrier()

    def prepare_network(self, half: int) -> None:
        """The weight set from ``half`` in FP16 for the cooperative-vector kernels, gradient slots cleared.
        Records while steps are recorded, else runs and waits."""
        standalone = self.encoder is None
        if standalone:
            self.encoder = self.device.create_command_encoder()
        try:
            self.encoder.convert_coop_vec_matrices(self.buffers['coop_weights'], self.coop.matrices,
                                                   self.buffers['weights_q'],
                                                   [self.coop.source(l, half) for l in range(len(LAYER_SHAPES))])
            self.encoder.clear_buffer(self.buffers['grad_slots'])
            self.dispatch(self.module.ntc_pack_biases, self.state_fields, half, self.buffers['coop_biases'],
                          spy.grid(((NUM_PARAMS - NUM_WEIGHTS) // 2,)))
        finally:
            if standalone:
                encoder, self.encoder = self.encoder, None
                self.device.submit_command_buffer(encoder.finish())
                self.device.wait()

    def iteration_end(self) -> int:
        """Step count at the end of the iteration the next step belongs to."""
        return min((self.step // self.s.steps_per_iteration + 1) * self.s.steps_per_iteration, self.s.steps)

    def record_step(self) -> None:
        """Records the next step into ``self.encoder`` and advances the host state. At the end of an iteration
        also copies the step's texel losses for ``read_loss``."""
        s, m, step, end = self.s, self.module, self.step, self.iteration_end()
        freeze_step, fp8_step, int8_step = self.schedule()
        quantize_fp8 = fp8_step < step <= int8_step
        quantize_int8 = step > int8_step
        half = NUM_PARAMS if quantize_int8 else 0

        if step == fp8_step:
            for name in ('weights_base', 'weights_q'):
                self.encoder.copy_buffer(self.buffers[name], NUM_PARAMS * 4, self.buffers[name], 0, NUM_PARAMS * 4)
            self.encoder.global_barrier()
        if quantize_fp8 or quantize_int8:
            self.dispatch(m.ntc_fake_quantize, self.state_fields, half, int(quantize_fp8), spy.grid((NUM_COLUMNS,)))
        self.prepare_network(half)

        self.dispatch(m.ntc_clear_mask, self.state_fields, spy.grid(((self.total_pixels + 31) // 32,)))
        stage = STAGE_INT8 if quantize_int8 else STAGE_FP8
        loss_params = {'gradient_scale': float(F32(self.loss_scale) / F32(s.batch * s.micro_batches)),
                       'accumulate_latents': int(step < freeze_step and s.train_latents)}
        network = self.network(stage, half)
        for micro in range(s.micro_batches):
            if micro:
                self.encoder.clear_buffer(self.buffers['grad_slots'])
                self.encoder.global_barrier()
            self.sampling['seed'] = self.rng.next_uint32()
            self.dispatch(m.ntc_sample, self.sampling, spy.grid((s.batch,)))
            if self.bsdf is None:
                self.dispatch(m.ntc_train_texel, network, self.state_fields, loss_params, spy.grid((s.batch,)))
            else:
                self.bsdf.prepare(network, self.state_fields, self.dispatch)
                self.bsdf.train(network, self.state_fields, loss_params, self.control_variate, self.dispatch)
            self.dispatch(m.ntc_weight_chunk, self.state_fields, spy.grid((s.batch // s.chunk, NUM_PARAMS)))
            self.dispatch(m.ntc_weight_sum, self.state_fields, int(micro > 0), spy.grid((NUM_PARAMS,)))

        network_lr = cosine(step, s.network_learning_rate, s.steps)
        self.dispatch(m.ntc_network_adam, self.state_fields,
                      adam_constants(s, self.loss_scale, step + 1, network_lr, self.rng.next_uint32(), half),
                      spy.grid((NUM_PARAMS,)))
        if step == freeze_step:
            self.dispatch(m.ntc_freeze, self.state_fields, spy.grid((self.total_pixels * self.num_features,)))
        elif step < freeze_step and s.train_latents:
            grid_lr = cosine(step, s.grid_learning_rate, s.steps)
            self.dispatch(m.ntc_latent_adam, self.network(stage, half), self.state_fields,
                          adam_constants(s, self.loss_scale, step + 1, grid_lr, self.rng.next_uint32(), 0),
                          spy.grid((self.num_features // 2, self.total_pixels)))

        self.step += 1
        if self.step == end:
            self.encoder.copy_buffer(self.loss_readback, 0, self.buffers['texel_loss'], 0, self.loss_readback.size)

    def read_loss(self) -> float:
        """Loss of the last step of the last finished iteration; the device must have finished it."""
        loss = float(self.loss_readback.to_numpy().view(F32).astype(np.float64).sum()) / self.s.batch
        divisor = self.s.valid_channels if self.bsdf is None else self.bsdf.loss_divisor
        return float(F32(loss) / F32(divisor))

    def submit(self, steps: int) -> int:
        """Records ``steps`` steps into one command buffer and submits it. Returns its fence value."""
        self.encoder = self.device.create_command_encoder()
        try:
            for _ in range(steps):
                self.record_step()
        finally:
            encoder, self.encoder = self.encoder, None
        self.fence_value += 1
        self.device.submit_command_buffers([encoder.finish()], signal_fences=[self.fence],
                                           signal_fence_values=[self.fence_value])
        return self.fence_value

    def run_step(self) -> float | None:
        """One training step, waited for. Returns the loss when it is the last step of an iteration."""
        end = self.iteration_end()
        self.fence.wait(self.submit(1))
        return self.read_loss() if self.step == end else None

    def update_loss_scale(self, loss: float) -> None:
        """SDK loss scale for the next iteration from the loss of the last one."""
        self.loss_scale = float(min(F32(32768), F32(128) / np.sqrt(F32(loss)))) if loss > 0 else 32768.0

    def finish_iteration(self, start: float, steps: int, on_iteration: Callable[[Iteration], None] | None) -> None:
        loss = self.read_loss()
        item = Iteration(self.step, loss, self.loss_scale, (time.perf_counter() - start) * 1000 / steps)
        self.update_loss_scale(loss)
        self.history.append(item)
        if on_iteration:
            on_iteration(item)

    def run(self, on_iteration: Callable[[Iteration], None] | None = None) -> list[Iteration]:
        """Steps in blocks of ``block_steps`` batches per command buffer, up to ``blocks_in_flight`` queued; the
        device is waited for only at the end of each iteration, where the loss scale changes."""
        block = max(1, self.s.block_steps // self.s.micro_batches)
        while self.step < self.s.steps:
            start, first, end = time.perf_counter(), self.step, self.iteration_end()
            in_flight: list[int] = []
            while self.step < end:
                if len(in_flight) == self.s.blocks_in_flight:
                    self.fence.wait(in_flight.pop(0))
                in_flight.append(self.submit(min(block, end - self.step)))
            self.fence.wait(in_flight[-1])
            self.finish_iteration(start, end - first, on_iteration)
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
