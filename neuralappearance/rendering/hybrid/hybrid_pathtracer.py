# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid path tracer with pluggable material evaluation.

Megakernel mode evaluates every material and bounce in one GPU dispatch.
Wavefront mode dispatches intersection and propagation separately. Between
them, a ``ScatterDataGenerator`` evaluates requests for the target material;
other materials remain inline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import falcor2 as f2
import numpy as np
import slangpy as spy
from falcor2.editor.scene_shader import SceneShaderHelper

if TYPE_CHECKING:
    from datagen.scatter_data_generator import ScatterDataGenerator


_HYBRID_MODULE_NAME = 'rendering.hybrid.hybrid_path_tracer'


# These kernel objects are bound once per render pass.
@dataclass
class _BoundKernels:
    """Bound kernel objects for a single render pass."""

    generate_path_state: Any
    clear_radiance: Any
    intersect_compact_gather: Any
    scatter_propagate: Any
    # This callable is present only in megakernel mode.
    megakernel: Any | None = None


class HybridPathTracer:
    """Hybrid path tracer with pluggable material scatter evaluation.

    ``set_target_material()`` selects the render mode. A scatter generator
    enables wavefront mode; omitting it selects megakernel mode.

    The caller must assign materials and call ``scene.update()`` before
    ``render()``. ``prepare_scene_material()`` performs this setup for the
    repository's rendering paths.
    """

    def __init__(self, device: spy.Device):
        self._device = device

        # Load ``HybridPathTracer`` once, without linking, for type reflection.
        # ``SceneShaderHelper`` creates scene-linked modules lazily.
        self._base_device_module = device.load_module('rendering/hybrid/hybrid_path_tracer.slang')
        self._base_module = spy.Module(self._base_device_module)

        self._scene_shader = SceneShaderHelper(device)
        # ``_prepare()`` creates the linked module.
        self._module: spy.Module | None = None

        # Mutable configuration -- property setters mark _dirty = True.
        self._enable_nee = True
        self._enable_mis = True
        self._enable_emissive_triangles = True
        self._enable_env_map = True
        self._max_depth = 10
        self._scene: f2.Scene | None = None
        self._scatter_generator: ScatterDataGenerator | None = None
        self._target_material_id: int | None = None
        self._target_material_resolution: int | None = None
        self._dirty = True
        # Track the mode used for the current buffer allocation.
        self._buffers_wavefront: bool | None = None

        self.profile = False

        # Profiling accumulators; used when self.profile is enabled.
        self._perf: dict[str, float] = {}
        self._perf_counts: dict[str, int] = {}
        self._perf_t0 = 0.0

        # ``_prepare()`` binds these kernels.
        self._bound: _BoundKernels | None = None

        # These types come from ``hybrid_path_tracer.slang`` and remain stable
        # when the scene module is relinked.
        self._eval_req_type = self._base_module.layout.require_type_by_name('ScatterEventQuery')
        self._eval_request_stride: int = self._eval_req_type.buffer_layout.reflection.size
        self._sample_type = self._base_module.layout.require_type_by_name('ScatterEventResult')
        self._path_state_type = self._base_module.layout.require_type_by_name('HybridPathState')

        # Persistent GPU buffers (allocated lazily on first render).
        self._buffers_shape: tuple[int, int] | None = None
        self._path_state: spy.Tensor | None = None
        self._eval_samples: spy.Tensor | None = None
        self._compact_indices: spy.Tensor | None = None
        self._compact_requests: spy.Tensor | None = None
        self._compact_results: spy.Tensor | None = None
        self._reverse_compact: spy.Tensor | None = None
        self._counters: spy.Tensor | None = None

    def set_target_material(
        self,
        material_id: int | None,
        target_material_resolution: int,
        generator: ScatterDataGenerator | None = None,
    ) -> None:
        """Configure the target material for external scatter evaluation.

        When material_id is not None, a generator must be provided.

        When material_id is None, the path tracer runs in megakernel mode
        and the generator is ignored.

        The target_material_texture_resolution is required for non-neural
        materials.

        Sets _dirty = True, triggering kernel recompilation on the next
        render() call. Avoid calling more than necessary.

        The caller is responsible for calling prepare_scene_material()
        before calling render().
        """
        if material_id is not None and generator is None:
            raise ValueError('generator is required when material_id is provided.')
        self._scatter_generator = generator if material_id is not None else None
        self._target_material_id = material_id
        self._target_material_resolution = target_material_resolution
        self._dirty = True

    # Properties. Each sets the dirty flag on set.

    @property
    def scene(self) -> f2.Scene | None:
        return self._scene

    @scene.setter
    def scene(self, value: f2.Scene) -> None:
        self._scene = value
        self._dirty = True

    @property
    def scatter_generator(self) -> ScatterDataGenerator | None:
        return self._scatter_generator

    @property
    def target_material_id(self) -> int | None:
        """The falcor2 material ID being intercepted.

        Set via set_target_material(). Returns None in megakernel mode.
        """
        return self._target_material_id

    @property
    def wavefront_mode(self) -> bool:
        """Return whether a scatter generator selected wavefront mode."""
        return self._scatter_generator is not None

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._max_depth = value
        self._dirty = True

    @property
    def enable_nee(self) -> bool:
        return self._enable_nee

    @enable_nee.setter
    def enable_nee(self, value: bool) -> None:
        self._enable_nee = value
        self._dirty = True

    @property
    def enable_mis(self) -> bool:
        return self._enable_mis

    @enable_mis.setter
    def enable_mis(self, value: bool) -> None:
        self._enable_mis = value
        self._dirty = True

    @property
    def enable_emissive_triangles(self) -> bool:
        return self._enable_emissive_triangles

    @enable_emissive_triangles.setter
    def enable_emissive_triangles(self, value: bool) -> None:
        self._enable_emissive_triangles = value
        self._dirty = True

    @property
    def enable_env_map(self) -> bool:
        return self._enable_env_map

    @enable_env_map.setter
    def enable_env_map(self, value: bool) -> None:
        self._enable_env_map = value
        self._dirty = True

    @property
    def eval_req_type(self):
        """The reflected ScatterEventQuery Slang type."""
        return self._eval_req_type

    # Profiling helpers.

    def _perf_begin(self) -> None:
        if self.profile:
            self._device.wait()
            self._perf_t0 = time.perf_counter()

    def _perf_end(self, name: str) -> None:
        if self.profile:
            self._device.wait()
            elapsed = time.perf_counter() - self._perf_t0
            self._perf[name] = self._perf.get(name, 0.0) + elapsed
            self._perf_counts[name] = self._perf_counts.get(name, 0) + 1

    def perf_reset(self) -> None:
        self._perf.clear()
        self._perf_counts.clear()

    def perf_report(self) -> str:
        if not self._perf:
            return '(no profiling data)'
        total = sum(self._perf.values())
        lines = []
        for name, t in sorted(self._perf.items(), key=lambda x: -x[1]):
            count = self._perf_counts[name]
            per_call = t / count * 1000
            pct = t / total * 100
            lines.append(
                f'  {name:20s}: {t:7.3f}s  ({count:5d} calls, {per_call:6.2f}ms/call, {pct:5.1f}%)'
            )
        lines.append(f'  {"TOTAL":20s}: {total:7.3f}s')
        return '\n'.join(lines)

    # Internal helpers.

    def _get_constants(self) -> dict:
        return {
            'ENABLE_NEE': self._enable_nee,
            'ENABLE_MIS': self._enable_mis,
            'ENABLE_EMISSIVE_TRIANGLES': self._enable_emissive_triangles,
            'ENABLE_ENV_MAP': self._enable_env_map,
            'MAX_DEPTH': self._max_depth,
        }

    def _ensure_buffers(self, height: int, width: int) -> None:
        """Allocate (or re-allocate) persistent GPU buffers.

        All buffers are wavefront-only.  In megakernel mode they are freed
        immediately (set to None) so the GPU memory is available for other uses.
        The _buffers_wavefront flag is tracked alongside shape so that
        switching from megakernel back to wavefront at the same resolution
        correctly triggers reallocation.
        """
        shape = (height, width)
        wavefront = self.wavefront_mode
        if self._buffers_shape == shape and self._buffers_wavefront == wavefront:
            return

        self._buffers_shape = shape
        self._buffers_wavefront = wavefront

        if not wavefront:
            self._path_state = None
            self._eval_samples = None
            self._compact_indices = None
            self._compact_requests = None
            self._compact_results = None
            self._reverse_compact = None
            self._counters = None
            return

        n_pixels = height * width

        self._path_state = spy.Tensor.empty(self._device, shape, self._path_state_type)

        self._eval_samples = spy.Tensor.empty(self._device, shape, self._sample_type)

        # Reserve one compact index per pixel for the worst case in which every
        # path requires external evaluation.
        self._compact_indices = spy.Tensor.empty(self._device, (n_pixels,), 'uint')

        # Fused intersection writes requests for the scatter generator.
        self._compact_requests = spy.Tensor.empty(self._device, (n_pixels,), self._eval_req_type)

        # Bind a one-element placeholder when ``n_eval == 0`` because
        # ``scatter_propagate`` still requires a valid buffer.
        self._compact_results = spy.Tensor.empty(self._device, (1,), self._sample_type)

        # Map pixel indices back to compact thread indices. Fused intersection
        # writes the mapping and fused propagation reads it.
        self._reverse_compact = spy.Tensor.empty(self._device, (n_pixels,), 'uint')

        # The two counters contain ``n_alive`` and ``n_evaluate``.
        # Zeroed by the Python bounce loop before each fused intersect dispatch.
        self._counters = spy.Tensor.empty(self._device, (2,), 'uint')

    def _bind_scene(self, cursor: spy.ShaderCursor) -> None:
        self._scene_shader.bind_scene(cursor)

    def _bind(self, kernel_name: str, constants: dict):
        """Bind constants, conformances, and scene data to one kernel."""
        assert self._module is not None
        assert self._scene is not None
        return (
            getattr(self._module, kernel_name)
            .constants(constants)
            .type_conformances(self._scene.requirements.type_conformances)
            .write(self._bind_scene)
        )

    def _prepare(self, height: int, width: int) -> None:
        """Bind kernels for a render pass. Called automatically by render()."""
        if self._scene is None:
            raise RuntimeError('Set .scene before calling render()')

        wavefront_mode = self.wavefront_mode

        # Acquire a scene-linked module cached by ``SceneShaderHelper`` on
        # scene.requirements identity).
        self._module = self._scene_shader.get_module(self._scene, _HYBRID_MODULE_NAME)
        self._ensure_buffers(height, width)

        constants = self._get_constants()
        constants['WAVEFRONT_MODE'] = wavefront_mode
        # Always define ``TARGET_MATERIAL_ID`` because both specializations
        # reference the constant.
        constants['TARGET_MATERIAL_ID'] = (
            self._target_material_id if self._target_material_id is not None else 0
        )
        constants['TARGET_MATERIAL_RESOLUTION'] = (
            self._target_material_resolution if self._target_material_resolution is not None else 0
        )

        # Build megakernel only when NOT using external scatter generator.
        # In wavefront mode, omit the megakernel so ``_render_one_sample()``
        # selects the wavefront path.
        if not wavefront_mode:
            megakernel = self._bind_material_system_megakernel(constants)
        else:
            megakernel = None

        self._bound = _BoundKernels(
            generate_path_state=self._bind('generate_path_state', constants),
            clear_radiance=self._bind('clear_radiance', constants),
            intersect_compact_gather=self._bind(
                'wavefront_intersect_compact_gather',
                constants,
            ),
            scatter_propagate=self._bind(
                'wavefront_scatter_propagate',
                constants,
            ),
            megakernel=megakernel,
        )
        self._dirty = False

    def _bind_material_system_megakernel(self, constants: dict) -> Any:
        """Bind the material-system megakernel (no scatter generator needed)."""
        assert self._module is not None
        assert self._scene is not None
        kernel = (
            self._module.render_megakernel_scene.constants(constants)
            .type_conformances(self._scene.requirements.type_conformances)
            .write(self._bind_scene)
        )
        assert self._buffers_shape is not None
        shape = self._buffers_shape

        def dispatch(camera, color, iteration, fixed_mip_level, render_neural_material, command_encoder):
            args = (camera, color, iteration, fixed_mip_level, render_neural_material)
            if command_encoder is None:
                kernel.call(*args, tid=spy.grid(shape))
            else:
                kernel.append_to(command_encoder, *args, tid=spy.grid(shape))

        return dispatch

    def _render_one_sample(
        self,
        camera: f2.Camera,
        color: spy.Tensor,
        iteration: int,
        fixed_mip_level: int,
        render_neural_material: bool,
        command_encoder: spy.CommandEncoder | None = None,
    ) -> None:
        """Render one sample using pre-bound kernels."""
        assert self._bound is not None
        assert self._buffers_shape is not None
        bound = self._bound

        if bound.megakernel is not None:
            # Single dispatch -- entire bounce loop on GPU.
            self._perf_begin()
            bound.megakernel(
                camera, color, iteration, fixed_mip_level, render_neural_material, command_encoder
            )
            self._perf_end('megakernel')
            return

        if command_encoder is not None:
            raise ValueError('A command encoder is supported in megakernel mode only.')

        if any(
            value is None
            for value in (
                self._path_state,
                self._eval_samples,
                self._compact_indices,
                self._compact_requests,
                self._compact_results,
                self._reverse_compact,
                self._counters,
                self._scatter_generator,
            )
        ):
            raise RuntimeError('Wavefront rendering state is not initialized.')

        # Wavefront rendering generates primary rays once. Each bounce then
        # intersects and compacts requests, reads the counters, evaluates target
        # materials, and propagates the resulting paths. ``color`` doubles as
        # the float4 radiance accumulator.

        # Generate primary camera rays.
        self._perf_begin()
        bound.generate_path_state.call(
            camera,
            iteration,
            fixed_mip_level,
            render_neural_material,
            tid=spy.grid(self._buffers_shape),
            _result=self._path_state,
        )
        self._perf_end('generate_path_state')

        # Clear with ``w=1`` so the GPU output is immediately valid.
        self._perf_begin()
        bound.clear_radiance.call(color)
        self._perf_end('clear_radiance')

        # Trace and shade each bounce.
        for depth in range(self._max_depth):
            # Fused intersect + compact + gather.
            self._perf_begin()
            self._counters.copy_from_numpy(np.zeros(2, dtype=np.uint32))
            bound.intersect_compact_gather.call(
                self._path_state,
                color,
                self._eval_samples,
                self._compact_requests,
                self._compact_indices,
                self._reverse_compact,
                self._counters.storage,
                spy.call_id(),
                self._buffers_shape[1],
            )
            self._perf_end('intersect+compact+gather')

            # Sync: download counters.
            self._perf_begin()
            n_alive, n_evaluate = self._counters.to_numpy()
            self._perf_end('sync')

            if n_alive == 0:
                break

            # Evaluate BSDF at eval-pending hit points.
            if n_evaluate > 0:
                n_eval = int(n_evaluate)
                self._perf_begin()
                compact_req_view = self._compact_requests.view(shape=(n_eval,))
                results = self._scatter_generator.eval_scatter(
                    compact_req_view,
                    n_eval,
                    self._compact_indices,
                    iteration,
                    depth + 1,
                )
                self._perf_end('scatter_gen')
            else:
                results = self._compact_results

            # Fused scatter + propagate.
            self._perf_begin()
            bound.scatter_propagate.call(
                self._path_state,
                color,
                self._eval_samples,
                results,
                self._reverse_compact,
                spy.call_id(),
                self._buffers_shape[1],
            )
            self._perf_end('scatter+propagate')

    # Public API.

    def render(
        self,
        camera: f2.Camera,
        color: spy.Tensor,
        iteration: int,
        fixed_mip_level: int = 0,
        render_neural_material: bool = False,
        command_encoder: spy.CommandEncoder | None = None,
    ) -> None:
        """Render one sample, preparing kernels when configuration changes.

        Set ``scene`` first and use ``set_target_material()`` to configure the
        mode. Other property changes take effect on the next ``render()`` call.
        With ``command_encoder`` (megakernel mode only) the dispatch is appended
        to it instead of submitted; the caller submits it.
        """
        height, width = color.shape[0], color.shape[1]

        # Resolution change triggers re-prepare. Scene-requirement changes are
        # absorbed by ``SceneShaderHelper`` and its identity-cached module.
        if (height, width) != self._buffers_shape:
            self._dirty = True

        if self._dirty:
            self._prepare(height, width)

        camera.width = width
        camera.height = height
        camera.recompute()

        self._render_one_sample(
            camera, color, iteration, fixed_mip_level, render_neural_material, command_encoder
        )
