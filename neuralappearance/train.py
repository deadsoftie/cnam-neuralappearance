# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level orchestration for the sequential neural material training phases.

Each phase supplies its own data generator and Slang training kernel to a shared
optimization loop. That loop batches multiple iterations per GPU submission,
tracks competing model instances, reports completed work, and synchronizes at
schedule changes and checkpoints.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import commentjson as json
import neuralnetworks as nn
import numpy as np
import slangpy as spy
import util
from checkpoint import evaluate_validation_losses, save_checkpoint, save_loss_plots
from datagen import (
    DataGenerators,
    FalcorBsdfDataGenerator,
    ReferenceMaterials,
    SamplerDataGenerator,
    create_aux_data_generator,
)
from model import NeuralModel
from rendering import Renderer
from rendering.pathtracer_helpers import create_testscene
from slangpy.core.native import Shape
from training import (
    BatchInstanceScheduler,
    CosineAnnealingLRScheduler,
    LossBuffer,
    LossFunction,
    LRSchedulerChain,
    TrainingPhase,
    TrainingTargets,
)
from util import LossSeries, Timer

if TYPE_CHECKING:
    from datagen import DataGenerator
    from training import LRScheduler

# Verify that SlangPy comes from the external folder rather than an installed
# package.
import falcor2

falcor2_root = Path(falcor2.__file__).resolve().parent.parent
expected_slangpy_path = str(falcor2_root / 'external/slangpy')
actual_slangpy_path = str(Path(spy.__file__).parent)
if not actual_slangpy_path.startswith(expected_slangpy_path):
    print(
        f'WARNING: SlangPy loaded from wrong location: {actual_slangpy_path}, expected: {expected_slangpy_path}\n'
        'This is likely due to an externally installed SlangPy package in the current environment, rather than '
        "the one installed by running falcor2's setup.bat/sh."
    )


# Enable this flag to use ``print`` from inside Slang shaders.
enable_debug_prints = False

# Enable this flag to get a lot of logging from SlangPy about kernel runs.
enable_slangpy_debug_logging = False


# Encoding and direct optimization share one continuous BSDF axis in loss plots,
# while ``job_iteration`` counts completed phases for checkpoint numbering.
LossAxis = Literal['Bsdf', 'Sampler', 'Aux']


class TrainingPhaseState:
    """Translate phase-local iterations into job and loss-series coordinates."""

    def __init__(self, job: TrainingJob, phase: TrainingPhase, detail: str | None = None):
        self.job = job
        self.phase: TrainingPhase = phase
        self.detail = detail
        self.loss_axis: LossAxis = {
            'BsdfEncoding': 'Bsdf',
            'BsdfDirectOptimization': 'Bsdf',
            'Sampler': 'Sampler',
            'Aux': 'Aux',
        }[phase]

    def job_iteration(self, local_iteration: int) -> int:
        return self.job.job_iteration_offset + local_iteration

    def loss_iteration(self, local_iteration: int) -> int:
        return self.job.loss_iteration_offsets[self.loss_axis] + local_iteration

    def advance(self, num_iterations: int) -> None:
        self.job.advance_phase(self.loss_axis, num_iterations)

    @property
    def display_label(self) -> str:
        label = self.phase
        if self.detail is not None:
            label += f' ({self.detail})'
        return label


class InFlightIterationBlock:
    """
    This class makes it easier to decouple CPU from GPU work during training.
    It represents one block of optimizer iterations submitted to the GPU, and
    encapsulates most of the timing/wait work that needs to be done.

    To make CPU readback of buffers efficient (e.g. the loss buffer), this keeps
    a copy that should be read later and enqueues a copy from the device local
    buffer to the host visible one. We can keep submitting new work in the
    meantime. After the block completes, the host visible loss buffer can be
    read without stalling the GPU.
    """

    def __init__(self, device: spy.Device, device_loss_buffer: LossBuffer):
        self.device = device
        self.fence = device.create_fence(0)
        self.fence_val = 0

        # This command buffer captures the timestamp at the start of the frame.
        self.timestamp_cmd: spy.CommandEncoder

        # This command buffer records the training work.
        self.train_cmd: spy.CommandEncoder

        # Reference to the device local loss buffer.
        self.device_loss_buffer = device_loss_buffer
        # Host visible buffer that we will copy the losses into.
        self.loss_buffer = LossBuffer(device, device_loss_buffer.tensor.shape, readback=True)
        self.end_iteration = 0

        # GPU timers measure execution time without serializing every kernel.
        self.num_queries = 512
        self.query_pool = device.create_query_pool(spy.QueryType.timestamp, self.num_queries)
        # Each value is the time in seconds from the start of the block until a
        # query is reached.
        self.time_until_query: list[float] = []

        # CPU timers for completing one block (i.e. all CPU work) and building
        # the training command buffer (i.e. all the SlangPy work).
        self.epoch_timer = Timer(history=1)
        self.slangpy_timer = Timer(history=1)

        self.reset()

    def reset(self):
        self.query_index = 0
        self.timing_spans = {}
        self.force_log = False

    @contextmanager
    def gpu_timer_slangpy(self, name: str):
        """Context manager for timing SlangPy kernel calls.

        Example usage:
            with iteration_block.gpu_timer_slangpy('Descriptive String'):
                # SlangPy kernel calls to be timed.
        """

        def submit_query():
            # Insert a query during training. After the block completes,
            # Record the time from the start of the block until the GPU reaches
            # this query.
            if self.query_index >= self.num_queries:
                raise ValueError('Exceeding number of available queries')
            self.train_cmd.write_timestamp(self.query_pool, self.query_index)
            self.query_index += 1
            return self.query_index - 1

        if name not in self.timing_spans:
            self.timing_spans[name] = []
        start_index = submit_query()
        try:
            yield
        finally:
            end_index = submit_query()
            self.timing_spans[name].append((start_index, end_index))

    def get_timings(self):
        self.wait_for_completion()
        timestamps = self.query_pool.get_timestamp_results(0, self.query_index)
        num_iterations = 0
        timings = {}
        for key in self.timing_spans:
            indices = self.timing_spans[key]
            elapsed_total = sum([timestamps[end] - timestamps[start] for start, end in indices])
            timings[key] = elapsed_total
            # Most spans occur once per optimizer iteration, while data
            # generation occurs only once. The maximum span count therefore
            # gives the number of iterations in the block.
            num_iterations = max(num_iterations, len(indices))
        return timings, num_iterations

    def start_timer(self):
        self.epoch_timer.start()

    def start_training(self):
        # Begin building training work for this block.
        self.train_cmd = self.device.create_command_encoder()
        self.slangpy_timer.start()
        self.device_loss_buffer.tensor.clear(self.train_cmd)

    def stop_training(self):
        # Finish the block by copying results to host-visible buffers and
        # submitting the command buffer.
        self.slangpy_timer.stop()
        self.loss_buffer.copy_from(self.device_loss_buffer, self.train_cmd)
        self.fence_val += 1
        self.device.submit_command_buffers(
            command_buffers=[self.train_cmd.finish()],
            signal_fences=[self.fence],
            signal_fence_values=[self.fence_val],
        )
        self.train_cmd = None

    def stop_timer(self, end_iteration: int):
        # Marks the final optimizer iteration included in this block.
        self.epoch_timer.stop()
        self.end_iteration = end_iteration

    def wait_for_completion(self):
        self.fence.wait(self.fence_val)


class TrainingPhaseRuntime:
    """Optimizer, buffers, schedules, and instance ranking for one phase.

    This state advances only when an optimizer step is queued. It also owns the
    current batch shape, so pruning or batch-size schedule changes can rebuild
    all dependent buffers together.
    """

    def __init__(
        self,
        config: dict,
        neural_model: NeuralModel,
        lr_scheduler: LRScheduler,
        batch_instance_scheduler: BatchInstanceScheduler,
        phase_state: TrainingPhaseState,
    ):
        self.neural_model = neural_model
        self.phase_state = phase_state
        self.training_phase: TrainingPhase = phase_state.phase

        self.iteration = 0
        self.latest_instance_losses = np.zeros(self.neural_model.num_instances)
        self.lr_scheduler = lr_scheduler
        self.batch_instance_scheduler = batch_instance_scheduler
        self.learning_rate = config['training']['optimizer']['learning_rate']
        self.gradient_scale = config['training']['optimizer']['gradient_scale']
        start_scale = self.lr_scheduler.get_scale(0)
        optim_config = config['training']['optimizer']['hyperparams']

        print('Use regularized Adam optimizer.')
        self.adam_optimizer = nn.RegularizedAdamOptimizer(
            start_scale * self.learning_rate,
            invalid_gradient_counter=nn.RegularizedAdamOptimizer.create_invalid_gradient_counter(
                self.neural_model.module.device
            ),
            **optim_config,
        )
        # Keep an FP32 copy of every half-precision parameter. Adam updates that
        # copy before converting it back to the FP16 value used by the model.
        # https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
        self.optimizer: nn.Optimizer = nn.FullPrecisionOptimizer(
            self.adam_optimizer,
            self.gradient_scale,
        )

        if config['training'].get('deterministic', False):
            self.optimizer = nn.DeterministicOptimizer(self.optimizer, 1)

        self.optimizer.initialize(
            self.neural_model.module,
            self.neural_model.active_training_models(),
        )

        # Each generated training batch feeds one optimizer iteration. Generate
        # several at once to amortize data-generation overhead.
        self.num_batches_per_generation = config['data_generation']['num_batches_per_generation']

        # Should each instance receive the same, or totally uncorrelated
        # training data?
        self.share_data_between_instances = config['data_generation'][
            'share_data_between_instances'
        ]

        # Store checkpointing information.
        self.ckpt_period = config['checkpoints']['period']
        if self.ckpt_period <= 0:
            self.ckpt_period = self.num_iterations

        self.sample_generator = util.UniformSampleGenerator(
            self.neural_model.module,
            (1,),
        )
        self.batch_size = 0
        self._resize_buffers()

    def _resize_buffers(self):
        batch_size = self._current_scheduled_batch_size()
        print(f'Resizing training batch size to {batch_size}')

        self.batch_size = batch_size
        self.loss_scale = self.gradient_scale / self.batch_size

        if isinstance(self.optimizer, nn.DeterministicOptimizer):
            self.optimizer.set_batch_size(self.batch_size)

        # Buffers used to extract the losses from the training kernel.
        launch_shape = (self.neural_model.num_instances, self.batch_size)
        self.loss_buffer = LossBuffer(
            self.neural_model.module.device,
            launch_shape,
        )
        # We need access to random numbers during training.
        self.sample_generator.reshape((self.neural_model.num_instances, self.batch_size))

        self.neural_model.write_to_buffer()

    def prune_or_resize_if_needed(self):
        scheduled_num_instances = self._current_scheduled_num_instances()
        if scheduled_num_instances < self.neural_model.num_instances:
            self.latest_instance_losses = self.neural_model.prune_instances(
                scheduled_num_instances,
                self.latest_instance_losses,
            )
            self.optimizer.prune_models(set(self.neural_model.active_training_models()))

        scheduled_batch_size = self._current_scheduled_batch_size()
        if scheduled_batch_size != self.batch_size:
            self._resize_buffers()

    def update_instance_losses(self, instance_losses: np.ndarray) -> None:
        assert len(instance_losses) == self.neural_model.num_instances
        self.latest_instance_losses = instance_losses

    def best_instance_index(self) -> int:
        return int(self.latest_instance_losses.argmin())

    def generate_training_data(
        self, data_generator: DataGenerator, _append_to: spy.CommandEncoder | None = None
    ):
        if self.share_data_between_instances:
            batch_count_instances = 1
        else:
            batch_count_instances = self.neural_model.num_instances

        batch_count = batch_count_instances * self.num_batches_per_generation

        data = data_generator.generate_training_data(
            self.iteration, batch_count, self.batch_size, _append_to=_append_to
        )

        # Shape:
        # Shape: ``[batch_count_instances * num_batches_per_generation, batch_size]``.
        # Reshape into:
        # Shape: ``[num_batches_per_generation, batch_count_instances, batch_size]``.
        data = data.view((self.num_batches_per_generation, batch_count_instances, self.batch_size))

        # Broadcast to shape we should return:
        # Shape: ``[batch_count, num_instances, batch_size]``.
        # I.e. for each batch, one set of samples per instance.
        data = data.broadcast_to(
            Shape(
                (self.num_batches_per_generation, self.neural_model.num_instances, self.batch_size)
            )
        )

        # If ``batch_count_instances == 1``, repeat data for each instance;
        # otherwise, it is unique per instance.
        return data

    def step(self, command_buffer: spy.CommandEncoder):
        """Run the optimizer step and update learning rate."""
        self.optimizer.step(command_buffer)
        command_buffer.global_barrier()
        # Update the learning rate.
        self.iteration = min(self.iteration + 1, self.num_iterations)
        scale = self.lr_scheduler.get_scale(self.iteration)
        self.adam_optimizer.learning_rate = self.learning_rate * scale

    def get_num_iterations_for_next_block(self) -> int:
        """Calculate how many optimizer iterations to put in the next block.

        This method determines how many iterations can be queued before the next
        checkpoint or the end of training, whichever comes first.

        Returns:
            int: The number of optimizer iterations to submit, limited by:
                - The configured block size
                - The number of iterations until the next checkpoint
                - The number of iterations remaining in the training process
        """
        next_checkpoint = (self.iteration // self.ckpt_period + 1) * self.ckpt_period
        num_iterations = min(self.num_batches_per_generation, next_checkpoint - self.iteration)
        num_iterations = min(num_iterations, self.num_iterations - self.iteration)
        return num_iterations

    def is_ckpt_iteration(self) -> bool:
        """Check if the current iteration is a checkpoint iteration."""
        return self.iteration % self.ckpt_period == 0

    def is_running(self) -> bool:
        """Check if the training is not done."""
        return self.iteration < self.num_iterations

    def is_done(self) -> bool:
        """Check if the training is done."""
        return not self.is_running()

    def is_key_iteration(self) -> bool:
        """Return whether queued work must drain before host-side changes."""
        return (
            self.is_ckpt_iteration()
            or self.is_done()
            or self._current_scheduled_num_instances() < self.neural_model.num_instances
            or self._current_scheduled_batch_size() != self.batch_size
        )

    @property
    def num_iterations(self) -> int:
        return self.lr_scheduler.num_iterations

    def _current_scheduled_num_instances(self) -> int:
        return self.batch_instance_scheduler.num_instances_at(
            self.iteration,
            self.num_iterations,
        )

    def _current_scheduled_batch_size(self) -> int:
        return self.batch_instance_scheduler.batch_size_at(
            self.iteration,
            self.num_iterations,
        )


class TrainingReporter:
    """Read completed loss buffers and update logs and instance rankings."""

    def __init__(self, loss_series: LossSeries):
        self.loss_series = loss_series
        self.last_log_iterations: dict[TrainingPhase, int] = {}

    def report_iteration_block(
        self,
        iteration_block: InFlightIterationBlock,
        phase_runtime: TrainingPhaseRuntime,
    ) -> None:
        phase = phase_runtime.training_phase
        last_log_iteration = self.last_log_iterations.get(phase, -1000)
        if (
            not iteration_block.force_log
            and last_log_iteration <= iteration_block.end_iteration < last_log_iteration + 1024
        ):
            return
        self.last_log_iterations[phase] = iteration_block.end_iteration

        neural_model = phase_runtime.neural_model
        phase_state = phase_runtime.phase_state

        # Collect timings.
        gpu_timings, num_iterations = iteration_block.get_timings()
        gpu_timings_str = '  '.join(
            f'{key}: {1e3 * value:4.2f}ms' for key, value in gpu_timings.items()
        )

        # Total time is datagen plus the total training time.
        total_elapsed_gpu = sum(list(gpu_timings.values()))
        total_elapsed_cpu = iteration_block.epoch_timer.elapsed()

        # Keep track of the training losses and throughput.
        msamples = (num_iterations * neural_model.num_instances * phase_runtime.batch_size) * 1e-6

        training_loss_and_avg = iteration_block.loss_buffer.tensor.to_numpy()
        training_loss_and_avg = training_loss_and_avg.reshape((neural_model.num_instances, -1, 2))
        training_loss_and_avg = training_loss_and_avg.mean(axis=1)
        training_losses = training_loss_and_avg[..., 0]
        loss_avg = training_loss_and_avg[..., 1] / num_iterations

        lmin = training_losses.min()
        lmax = training_losses.max()
        lstd = training_losses.std()

        # Format the loss summary.
        if neural_model.num_instances > 4:
            training_losses_str = f'min: {lmin:.4f}, max: {lmax:.4f}, std: {lstd:.4f}'
        else:
            training_losses_str = ', '.join(f'{loss:.4f}' for loss in training_losses)

        iter_end = iteration_block.end_iteration
        iter_start = iter_end - num_iterations + 1
        print(
            f'Iter {iter_start:05d}-{iter_end:05d} '
            + f'Throughput: {msamples / total_elapsed_gpu:.1f} MSamples/s \n'
            + f'    (Losses) Training ----------------->: {training_losses_str} \n'
            + f'    (GPU Timings) Total: {total_elapsed_gpu * 1e3:5.1f}ms '
            + gpu_timings_str
            + ' \n'
            + f'    (CPU Timings) Total: {total_elapsed_cpu * 1e3:5.1f}ms '
            + f'SlangPy: {iteration_block.slangpy_timer.elapsed() * 1e3:5.1f}ms '
        )

        phase_runtime.update_instance_losses(training_losses)

        loss_name = self._loss_name(neural_model, phase_state)
        for i in range(neural_model.num_instances):
            self.loss_series.add(
                f'{loss_name} #{neural_model.instance_ids[i]}',
                iteration_block.end_iteration,
                training_losses[i],
            )
            self.loss_series.add(
                f'{loss_name} (iteration-averaged) #{neural_model.instance_ids[i]}',
                iteration_block.end_iteration,
                loss_avg[i],
            )

        self.loss_series.flush()

    def _loss_name(
        self,
        neural_model: NeuralModel,
        phase_state: TrainingPhaseState,
    ) -> str:
        if phase_state.phase == 'Sampler':
            return 'Training Loss Sampler'
        if phase_state.phase == 'Aux':
            return 'Training Loss Aux'
        return 'Training Loss Bsdf'


class TrainingJob:
    """Own job-wide state and run the pipelined optimization loop.

    Phase functions configure trainable components, schedules, data generation,
    and a kernel callback. ``TrainingJob`` handles submission, readback,
    checkpoint synchronization, reporting, and summary metadata uniformly.
    """

    def __init__(
        self,
        neural_model: NeuralModel,
        data_generators: DataGenerators,
        renderer: Renderer,
        reference_materials: ReferenceMaterials,
        loss_series: LossSeries,
        outfolder: Path,
    ):
        self.neural_model = neural_model
        self.data_generators = data_generators
        self.renderer = renderer
        self.reference_materials = reference_materials
        self.loss_series = loss_series
        self.outfolder = outfolder
        self.reporter = TrainingReporter(loss_series)
        with (self.outfolder / 'summary.json').open() as f:
            self.summary: dict[str, Any] = json.load(f)
        self.job_iteration_offset = 0
        self.loss_iteration_offsets: dict[LossAxis, int] = {
            'Bsdf': 0,
            'Sampler': 0,
            'Aux': 0,
        }

    @property
    def module(self) -> spy.Module:
        return self.neural_model.module

    @property
    def config(self) -> dict:
        return self.neural_model.config

    @property
    def device(self) -> spy.Device:
        return self.module.device

    def begin_phase(self, phase: TrainingPhase, detail: str | None = None) -> TrainingPhaseState:
        return TrainingPhaseState(self, phase, detail)

    def advance_phase(self, loss_axis: LossAxis, num_iterations: int) -> None:
        self.job_iteration_offset += num_iterations
        self.loss_iteration_offsets[loss_axis] += num_iterations

    def run_phase_optimization(
        self,
        phase_state: TrainingPhaseState,
        lr_scheduler: LRScheduler,
        batch_instance_scheduler: BatchInstanceScheduler,
        data_generator: DataGenerator,
        run_train_kernel: Callable[
            [TrainingPhaseRuntime, spy.Tensor, InFlightIterationBlock], None
        ],
    ) -> None:
        """Run one phase until its learning-rate schedule completes."""

        phase_runtime = TrainingPhaseRuntime(
            self.config,
            self.neural_model,
            lr_scheduler,
            batch_instance_scheduler,
            phase_state,
        )

        # Current iteration block data. This is set to a non-null value if we
        # reclaims a prior completed block. It is ``None`` when a new block
        # object must be created.
        iteration_block: InFlightIterationBlock | None = None
        # Iteration blocks currently queued on the GPU.
        iteration_blocks_in_flight: list[InFlightIterationBlock] = []

        print(f'{phase_runtime.phase_state.display_label}...')
        print('==========================================')

        opt_start_time = time.perf_counter()
        ckpt_time = 0.0

        while phase_runtime.is_running():
            phase_runtime.prune_or_resize_if_needed()

            # Record and submit one block of optimizer iterations.
            self._submit_iteration_block(
                phase_runtime,
                iteration_block,
                iteration_blocks_in_flight,
                data_generator,
                run_train_kernel,
            )

            # Reclaim finished blocks so their readback buffers can be reported
            # and reused.
            iteration_block = self._drain_inflight_iteration_blocks(
                phase_runtime,
                iteration_blocks_in_flight,
            )

            ckpt_time += self._maybe_save_checkpoint(
                phase_runtime,
            )

            if enable_debug_prints:
                self.device.flush_print()

            if phase_runtime.is_done():
                self.device.wait_for_idle()

        self._finish_optimization_phase(
            phase_runtime,
            opt_start_time,
            ckpt_time,
        )

        phase_runtime.phase_state.advance(phase_runtime.num_iterations)

    def _maybe_save_checkpoint(
        self,
        phase_runtime: TrainingPhaseRuntime,
    ) -> float:
        if phase_runtime.phase_state.phase not in self.config['checkpoints']['phases']:
            return 0.0
        if not (phase_runtime.is_ckpt_iteration() or phase_runtime.is_done()):
            return 0.0

        ckpt_start_time = time.perf_counter()
        save_checkpoint(
            self.module,
            self.config,
            self.neural_model,
            self.renderer,
            self.reference_materials,
            self.data_generators,
            self.outfolder,
            phase_runtime.phase_state.job_iteration(phase_runtime.iteration),
            best_instance_index=phase_runtime.best_instance_index(),
        )
        return time.perf_counter() - ckpt_start_time

    def _submit_iteration_block(
        self,
        phase_runtime: TrainingPhaseRuntime,
        iteration_block: InFlightIterationBlock | None,
        iteration_blocks_in_flight: list[InFlightIterationBlock],
        data_generator: DataGenerator,
        run_train_kernel: Callable[
            [TrainingPhaseRuntime, spy.Tensor, InFlightIterationBlock], None
        ],
    ) -> None:
        if (
            iteration_block is None
            or phase_runtime.batch_size != iteration_block.device_loss_buffer.tensor.shape[-1]
        ):
            iteration_block = InFlightIterationBlock(self.device, phase_runtime.loss_buffer)
        else:
            iteration_block.reset()

        iteration_block.start_timer()
        iteration_block.start_training()

        with iteration_block.gpu_timer_slangpy('Datagen'):
            training_data = phase_runtime.generate_training_data(
                data_generator, _append_to=iteration_block.train_cmd
            )

        for iteration_index in range(phase_runtime.get_num_iterations_for_next_block()):
            batch_index = iteration_index % training_data.shape[0]

            with iteration_block.gpu_timer_slangpy('Train kernel'):
                run_train_kernel(phase_runtime, training_data[batch_index], iteration_block)
                iteration_block.train_cmd.global_barrier()

            with iteration_block.gpu_timer_slangpy('Step kernel'):
                phase_runtime.step(iteration_block.train_cmd)

        iteration_block.stop_training()
        is_key_iteration = phase_runtime.is_key_iteration()
        iteration_block.stop_timer(
            phase_runtime.phase_state.loss_iteration(phase_runtime.iteration)
        )
        iteration_block.force_log = is_key_iteration
        iteration_blocks_in_flight.append(iteration_block)

    def _drain_inflight_iteration_blocks(
        self,
        phase_runtime: TrainingPhaseRuntime,
        iteration_blocks_in_flight: list[InFlightIterationBlock],
    ) -> InFlightIterationBlock | None:
        """Report completed blocks and return the last one for buffer reuse.

        Normal iterations allow several submissions to remain queued. A key
        iteration drains all of them before pruning, resizing, or checkpointing
        uses their final losses.
        """
        max_blocks_in_flight = 0 if phase_runtime.is_key_iteration() else 8
        iteration_block: InFlightIterationBlock | None = None
        while len(iteration_blocks_in_flight) > max_blocks_in_flight:
            iteration_block = iteration_blocks_in_flight.pop(0)
            iteration_block.wait_for_completion()

            self.reporter.report_iteration_block(iteration_block, phase_runtime)
        return iteration_block

    def _finish_optimization_phase(
        self,
        phase_runtime: TrainingPhaseRuntime,
        opt_start_time: float,
        ckpt_time: float,
    ) -> None:
        total_time = time.perf_counter() - opt_start_time
        optimization_time = total_time - ckpt_time
        iterations_per_second = phase_runtime.num_iterations / optimization_time
        invalid_gradients, invalid_gradients_seen = (
            phase_runtime.adam_optimizer.read_invalid_gradients()
        )
        phase_state = phase_runtime.phase_state
        print('------------------------------------------')
        print(f'{phase_state.display_label} done!')
        print(f'  * Total time: {total_time:.2f}s')
        print(f'  * Optimization time: {optimization_time:.2f}s')
        print(f'  * Iterations/second: {iterations_per_second:.2f}')
        print(f'  * Checkpoint time: {ckpt_time:.2f}s')
        print(f'  * Non-finite gradients: {invalid_gradients} (any: {invalid_gradients_seen})')
        print('------------------------------------------')

        self.summary['phases'].append(
            {
                'phase': phase_state.phase,
                'label': phase_state.display_label,
                'num_iterations': phase_runtime.num_iterations,
                'total_time': total_time,
                'optimization_time': optimization_time,
                'iterations_per_second': iterations_per_second,
                'checkpoint_time': ckpt_time,
                'invalid_gradients': invalid_gradients,
                'invalid_gradients_seen': invalid_gradients_seen,
            }
        )
        self._save_summary()

        save_loss_plots(self.config, self.loss_series)

    def record_bake_time(self, bake_time: float) -> None:
        """Record the wall time of baking the encoder output into the latents."""
        print(f'  * Bake time: {bake_time:.2f}s')
        self.summary['bake_time'] = bake_time
        self._save_summary()

    def _save_summary(self) -> None:
        with (self.outfolder / 'summary.json').open('w') as f:
            json.dump(self.summary, f, indent=2)


def training_main(args) -> None:
    """Create the shared job state and execute every configured phase."""

    # Setup training job.
    config = util.load_config(args.config)

    outfolder = util.initialize_job(args.outfolder, args.outfolder_policy, args.message, args.group)
    loss_series = LossSeries(outfolder)

    device = spy.Device(
        type=spy.DeviceType.vulkan,
        enable_print=enable_debug_prints,
        enable_debug_layers=False,
        enable_hot_reload=False,
        compiler_options={
            'include_paths': [
                spy.SHADER_PATH,
                Path(__file__).parent.absolute(),
                falcor2_root / 'slang',
                *nn.slang_include_paths(),
            ],
            'disable_warnings': [
                '41012',  # entry point uses additional capabilities that are not part of the specified profile
                '41018',  # returning without initializing out parameter
                '41021',  # default initializer will not initialize field
                '41035',  # possible use of uninitialized variable
            ],
        },
        bindless_options=spy.BindlessDesc(buffer_count=262144),
    )
    print(f'Using SlangPy device: {device.info.adapter_name}')

    module = spy.Module.load_from_file(device, 'training/train.slang')
    if enable_slangpy_debug_logging:
        module.logger = spy.Logger(level=spy.LogLevel.debug)

    # Keep one shared scene for data generation and reference renderings.
    # We'll create a separate scene for neural checkpoint renderings later.
    scene = create_testscene(device)
    reference_materials = ReferenceMaterials.create(
        device,
        config,
        args.assets_paths,
        scene=scene,
    )

    data_generators = DataGenerators(
        FalcorBsdfDataGenerator(
            device,
            config,
            reference_materials,
            'BsdfEncoding',
        )
    )

    # Create reference renderings.
    renderer = Renderer(device, config, reference_materials)
    renderer.render_reference_materials(
        config, reference_materials, data_generators, outfolder, TrainingTargets('bsdf')
    )
    # Setup the neural material model.
    neural_model = NeuralModel(module, config, reference_materials)
    job = TrainingJob(
        neural_model=neural_model,
        data_generators=data_generators,
        renderer=renderer,
        reference_materials=reference_materials,
        loss_series=loss_series,
        outfolder=outfolder,
    )

    training_bsdf_encoding(job)
    training_bsdf_diropt(job)
    training_sampler(job)
    training_aux(job)

    evaluate_validation_losses(
        config,
        neural_model,
        reference_materials,
        outfolder,
    )

    print('TRAINING COMPLETE!')

    util.finalize_job(args.outfolder)

    if args.view:
        print('\nLaunching visualizer...')
        from visualizer import NeuralMaterialVisualizer, find_latest_checkpoint

        vis = NeuralMaterialVisualizer(find_latest_checkpoint(outfolder), args.assets_paths)
        vis.main_loop()


def training_bsdf_encoding(job: TrainingJob) -> None:
    """Train the encoder and BSDF decoder, then bake the latents."""

    neural_model = job.neural_model
    module = job.module
    config = job.config
    phase_state = job.begin_phase('BsdfEncoding')

    print()

    # Check if the BSDF needs training.
    if neural_model.encoder is None:
        print('No BSDF encoder found, skipping "training_bsdf_encoding()".')
        return
    if neural_model.encoder.status == 'Done' and neural_model.decoder.status == 'Done':
        phase_state.advance(config['training']['num_iterations']['bsdf'])
        print('BSDF already trained, skipping "training_bsdf_encoding()".')
        return

    lr_scheduler = CosineAnnealingLRScheduler(
        start_scale=1.0,
        end_scale=min(1.0, config['training']['optimizer']['cosine_annealing_scale']),
        num_iterations=config['training']['num_iterations']['bsdf'],
    )
    batch_instance_scheduler = BatchInstanceScheduler(
        config['training']['instance_schedule'],
        config['data_generation']['batch_size_schedule'],
    )
    loss_function = LossFunction(
        config['training']['loss_function'],
        TrainingTargets('bsdf'),
    )

    neural_model.start_training(neural_model.encoder)
    neural_model.start_training(neural_model.decoder)

    train_bsdf_encoding = module.train_bsdf_encoding.as_func().map(
        encoder=(0,), decoder=(0,), sample=(0, 1)
    )

    def run_train_kernel(phase_runtime, sample, iteration_block):
        train_bsdf_encoding(
            encoder=neural_model.encoder_buffer,
            decoder=neural_model.decoder_buffer,
            sample=sample,
            loss_scale=phase_runtime.loss_scale,
            call_id=spy.call_id(),
            loss_buffer=phase_runtime.loss_buffer.tensor,
            loss_function=loss_function,
            _append_to=iteration_block.train_cmd,
        )

    job.run_phase_optimization(
        phase_state=phase_state,
        lr_scheduler=lr_scheduler,
        batch_instance_scheduler=batch_instance_scheduler,
        data_generator=job.data_generators.bsdf,
        run_train_kernel=run_train_kernel,
    )

    # Preserve the final encoder result in textures used by all later phases.
    assert len(neural_model.decoder.instances) == 1, (
        'Must have pruned to a single decoder instance by now.'
    )
    assert len(neural_model.encoder.instances) == 1, (
        'Must have pruned to a single encoder instance by now.'
    )
    bake_start_time = time.perf_counter()
    neural_model.latent_texture.instances[0].generate_from_encoder(
        module,
        job.data_generators,
        job.reference_materials,
        neural_model.num_mip_levels,
        neural_model.encoder.instances[0],
    )
    job.device.wait()
    job.record_bake_time(time.perf_counter() - bake_start_time)
    neural_model.decoder.status = 'Done'
    neural_model.encoder.status = 'Done'
    neural_model.latent_texture.status = 'Done'


def training_bsdf_diropt(job: TrainingJob) -> None:
    """Train latent texels and the BSDF decoder without an encoder in the path.

    This either initializes the representation directly or fine-tunes the
    encoded result produced by ``training_bsdf_encoding()``.
    """

    neural_model = job.neural_model
    module = job.module
    config = job.config
    phase_state = job.begin_phase('BsdfDirectOptimization')

    print()

    # Only relevant if there was first an encoding phase.
    encoder = neural_model.encoder
    finetuning = encoder is not None and encoder.status == 'Done'

    if finetuning:
        # We previously had an encoding phase, so let's try finetuning.
        num_finetuning_iterations = config['training']['num_iterations'].get('bsdf_finetuning', 0)
        if num_finetuning_iterations == 0:
            print('No finetuning iterations configured, skipping "training_bsdf_diropt()".')
            return

        num_warmup_iterations = min(1000, num_finetuning_iterations // 4)
        lr_scheduler = LRSchedulerChain(
            [
                CosineAnnealingLRScheduler(
                    start_scale=0.0,
                    end_scale=1.0,
                    num_iterations=num_warmup_iterations,
                ),
                CosineAnnealingLRScheduler(
                    start_scale=1.0,
                    end_scale=config['training']['optimizer']['cosine_annealing_scale'],
                    num_iterations=num_finetuning_iterations - num_warmup_iterations,
                ),
            ]
        )
    else:
        # No encoding phase, so we are doing direct optimization from the start.
        if neural_model.decoder.status == 'Done' and neural_model.latent_texture.status == 'Done':
            phase_state.advance(config['training']['num_iterations']['bsdf'])
            print('BSDF already trained, skipping "training_bsdf_diropt()".')
            return

        lr_scheduler = CosineAnnealingLRScheduler(
            start_scale=1.0,
            end_scale=min(1.0, config['training']['optimizer']['cosine_annealing_scale']),
            num_iterations=config['training']['num_iterations']['bsdf'],
        )

    configured_batch_instance_scheduler = BatchInstanceScheduler(
        config['training']['instance_schedule'],
        config['data_generation']['batch_size_schedule'],
    )
    batch_instance_scheduler = BatchInstanceScheduler(
        [neural_model.num_instances],
        [configured_batch_instance_scheduler.batch_sizes[-1]],
    )

    neural_model.start_training(neural_model.decoder)
    neural_model.start_training(neural_model.latent_texture)

    assert neural_model.num_instances == 1, (
        'Direct optimization of latents is not compatible with multi-instance training.'
    )
    assert not config['model']['latents']['normalize'], (
        'Direct optimization of latents is not compatible with latent normalization.'
    )

    job.data_generators.bsdf = FalcorBsdfDataGenerator(
        job.device,
        config,
        job.reference_materials,
        'BsdfDirectOptimization',
    )

    loss_function = LossFunction(
        config['training']['loss_function'],
        TrainingTargets('bsdf'),
    )

    if config['training']['deterministic']:
        print(
            'Warning: Deterministic optimization is likely to exceed available memory when using direct optimization of latent textures!'
        )

    train_bsdf_diropt = module.train_bsdf_diropt.as_func().map(
        latent_texture=(0,), decoder=(0,), sample=(0, 1), sg=(0, 1)
    )

    def run_train_kernel(phase_runtime, sample, iteration_block):
        train_bsdf_diropt(
            latent_texture=neural_model.latent_texture_buffer,
            decoder=neural_model.decoder_buffer,
            sample=sample,
            sg=phase_runtime.sample_generator,
            loss_scale=phase_runtime.loss_scale,
            call_id=spy.call_id(),
            loss_buffer=phase_runtime.loss_buffer.tensor,
            loss_function=loss_function,
            _append_to=iteration_block.train_cmd,
        )

    job.run_phase_optimization(
        phase_state=phase_state,
        lr_scheduler=lr_scheduler,
        batch_instance_scheduler=batch_instance_scheduler,
        data_generator=job.data_generators.bsdf,
        run_train_kernel=run_train_kernel,
    )

    # Mark neural model components as trained.
    neural_model.decoder.status = 'Done'
    neural_model.latent_texture.status = 'Done'


def training_sampler(job: TrainingJob) -> None:
    """Train the importance sampler against the frozen neural BSDF."""

    neural_model = job.neural_model
    module = job.module
    config = job.config
    phase_state = job.begin_phase('Sampler')

    print()

    if neural_model.sampler is None:
        print('No sampler configured, skipping "training_sampler()".')
        return
    num_sampler_iterations = config['training']['num_iterations'].get('sampler', 0)
    if num_sampler_iterations == 0:
        print('No sampler iterations configured, skipping "training_sampler()".')
        return
    if neural_model.sampler.status == 'Done':
        phase_state.advance(num_sampler_iterations)
        print('Sampler already trained, skipping "training_sampler()".')
        return

    lr_scheduler = CosineAnnealingLRScheduler(
        start_scale=1.0,
        end_scale=min(1.0, config['training']['optimizer']['cosine_annealing_scale']),
        num_iterations=num_sampler_iterations,
    )
    batch_instance_scheduler = BatchInstanceScheduler(
        config['training']['instance_schedule'],
        config['data_generation']['batch_size_schedule'],
    )

    # New data generator using the previously trained neural material.
    job.data_generators.sampler = SamplerDataGenerator(
        job.device,
        config,
        job.reference_materials,
        neural_model,
    )

    neural_model.start_training(neural_model.sampler)

    train_sampler = module.train_sampler.as_func().map(
        decoder=(0,), sampler=(0,), sample=(0, 1), sg=(0, 1)
    )

    def run_train_kernel(phase_runtime, sample, iteration_block):
        train_sampler(
            decoder=neural_model.decoder_buffer,
            sampler=neural_model.sampler_buffer,
            sample=sample,
            sg=phase_runtime.sample_generator,
            loss_scale=phase_runtime.loss_scale,
            call_id=spy.call_id(),
            loss_buffer=phase_runtime.loss_buffer.tensor,
            _append_to=iteration_block.train_cmd,
        )

    job.run_phase_optimization(
        phase_state=phase_state,
        lr_scheduler=lr_scheduler,
        batch_instance_scheduler=batch_instance_scheduler,
        data_generator=job.data_generators.sampler,
        run_train_kernel=run_train_kernel,
    )

    # Mark neural model components as trained.
    assert len(neural_model.sampler.instances) == 1, (
        'Must have pruned to a single sampler instance by now.'
    )
    neural_model.sampler.status = 'Done'


def training_aux(job: TrainingJob) -> None:
    """Train the auxiliary decoder on the frozen latent representation."""

    neural_model = job.neural_model
    module = job.module
    config = job.config
    phase_state = job.begin_phase('Aux')

    print()

    if neural_model.aux is None:
        print('No aux configured, skipping "training_aux()".')
        return
    num_aux_iterations = config['training']['num_iterations'].get('aux', 0)
    if num_aux_iterations == 0:
        print('No aux iterations configured, skipping "training_aux()".')
        return
    if neural_model.aux.status == 'Done':
        phase_state.advance(num_aux_iterations)
        print('Aux already trained, skipping "training_aux()".')
        return

    aux_targets = TrainingTargets(config['model']['aux']['targets'])

    lr_scheduler = CosineAnnealingLRScheduler(
        start_scale=1.0,
        end_scale=min(1.0, config['training']['optimizer']['cosine_annealing_scale']),
        num_iterations=num_aux_iterations,
    )
    batch_instance_scheduler = BatchInstanceScheduler(
        config['training']['instance_schedule'],
        config['data_generation']['batch_size_schedule'],
    )
    # Always use a relative L2 loss for the aux signals. Unlike the BSDF, they
    # may contain negative values (e.g. normals).
    aux_loss = {
        'type': 'RelativeL2',
        'epsilon': config['training']['loss_function'].get('epsilon', 1e-4),
    }
    loss_function = LossFunction(aux_loss, aux_targets)

    # Setup new data generator for the auxiliary outputs.
    job.data_generators.aux = create_aux_data_generator(
        job.device,
        config,
        job.reference_materials,
        aux_targets,
        neural_model,
    )

    # Render the aux references.
    job.renderer.render_reference_materials(
        config,
        job.reference_materials,
        job.data_generators,
        job.outfolder,
        aux_targets,
    )

    # Activate only the auxiliary decoder for this phase.
    neural_model.start_training(neural_model.aux)

    train_aux = module.train_aux.as_func().map(
        latent_texture=(0,), aux=(0,), sample=(0, 1), sg=(0, 1)
    )

    def run_train_kernel(phase_runtime, sample, iteration_block):
        train_aux(
            latent_texture=neural_model.latent_texture_buffer,
            aux=neural_model.aux_buffer,
            sample=sample,
            sg=phase_runtime.sample_generator,
            loss_scale=phase_runtime.loss_scale,
            call_id=spy.call_id(),
            loss_buffer=phase_runtime.loss_buffer.tensor,
            loss_function=loss_function,
            _append_to=iteration_block.train_cmd,
        )

    job.run_phase_optimization(
        phase_state=phase_state,
        lr_scheduler=lr_scheduler,
        batch_instance_scheduler=batch_instance_scheduler,
        data_generator=job.data_generators.aux,
        run_train_kernel=run_train_kernel,
    )

    # Mark neural model components as trained.
    assert len(neural_model.aux.instances) == 1, 'Must have pruned to a single aux instance by now.'
    neural_model.aux.status = 'Done'


if __name__ == '__main__':
    args = util.parse_args()
    spy.set_dump_generated_shaders(False)

    if any('debugpy' in module for module in sys.modules):
        # Call ``training_main`` directly when debugging so execution stops at
        # the original exception.
        training_main(args)
        print('Exiting normally from debug session.')
    else:
        try:
            training_main(args)
            print('Exiting normally.')
        except Exception as e:
            print(f'Error during training: {e}', file=sys.stderr)
            print(traceback.format_exc())
            sys.exit(1)
