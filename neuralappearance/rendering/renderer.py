# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import falcor2 as f2
import numpy as np
import slangpy as spy
from training import TrainingTargets
from util.math_helpers import udim_offset
from util.timer import Timer

from .hybrid.hybrid_pathtracer import HybridPathTracer
from .hybrid.scene_material_setup import prepare_scene_material
from .neural_material import NeuralMaterial
from .pathtracer_helpers import apply_udim_offset_to_scene, create_testscene, split_aux_channels

if TYPE_CHECKING:
    from checkpoint import Checkpoint
    from datagen import DataGenerators, ReferenceMaterial, ReferenceMaterials
    from model import NeuralModelCheckpoint

WIDTH = 1920
HEIGHT = 1080


def _get_scene_camera(scene: f2.Scene) -> f2.Camera:
    """Return the first f2.Camera component in the scene."""
    for component in scene.components:
        if isinstance(component, f2.Camera):
            return component
    raise RuntimeError('No Camera component found in scene')


class Renderer:
    """Checkpoint renderer backed by separate reference and neural scenes."""

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials | None = None,
    ):
        self.device = device
        rendering_config = config['checkpoints']['rendering']
        self.spp = rendering_config['spp']
        self.save_reference = rendering_config['save_reference']
        self.save_neural = rendering_config['save_neural']
        self.render_all_udims = rendering_config['all_udims']
        self.render_all_mips = True
        self.format_ext = rendering_config['format']
        self.latent_filter = rendering_config.get('latent_filter', 'stochastic')

        if reference_materials is None or reference_materials.f2_scene is None:
            raise RuntimeError(
                'Renderer requires a ReferenceMaterials with a pre-populated '
                'f2_scene. Build the scene via '
                'pathtracer_helpers.create_testscene(device) and pass it to '
                'ReferenceMaterials.create(..., scene=scene) before constructing '
                'the renderer.'
            )
        self.reference_scene = reference_materials.f2_scene
        # Keep generated reference modules out of neural checkpoint shaders.
        # The scene is created once and reused for every checkpoint.
        self.neural_scene = create_testscene(device)

        max_depth = rendering_config.get('max_depth', 10)
        self.reference_hybrid_pt = HybridPathTracer(device)
        self.reference_hybrid_pt.max_depth = max_depth
        self.reference_hybrid_pt.scene = self.reference_scene
        self.neural_hybrid_pt = HybridPathTracer(device)
        self.neural_hybrid_pt.max_depth = max_depth
        self.neural_hybrid_pt.scene = self.neural_scene

        self.reference_camera = _get_scene_camera(self.reference_scene)
        self.neural_camera = _get_scene_camera(self.neural_scene)
        for camera in (self.reference_camera, self.neural_camera):
            camera.width = WIDTH
            camera.height = HEIGHT
        self.output_tensor = spy.Tensor.empty(device, (HEIGHT, WIDTH), dtype='float4')

        # Load ``NeuralMaterialModel`` for ``eval_latent_texture()`` and
        # ``eval_aux()``.
        self._model_module = None

        # Aux accumulator module for the masked-add kernel inside the SPP loop.
        self._aux_accumulator_module = spy.Module.load_from_file(
            self.device,
            'rendering/hybrid/aux_accumulator.slang',
        )

        # The accumulator stores sum.xyz and sample count in .w across SPP.
        self._utils_module = spy.Module.load_from_file(device, 'falcor2/utils.slang')
        self._accumulator = spy.Tensor.empty(device, (HEIGHT, WIDTH), dtype='float4')

        # Cached hit-extraction kernel for aux rendering.
        self._hit_kernel: Any | None = None
        self._hit_kernel_key: tuple | None = None
        self._hit_kernel_base: Any | None = None

        # Neural materials are rebuilt per checkpoint so material payloads and
        # required modules stay immutable once the scene sees them.
        self._neural_model: NeuralModelCheckpoint | None = None
        self._neural_materials: dict[int, NeuralMaterial] = {}

    def render_reference_materials(
        self,
        config: dict,
        reference_materials: ReferenceMaterials,
        data_generators: DataGenerators,
        outfolder: Path,
        targets: TrainingTargets,
    ):
        if not self.save_reference:
            return

        print('Rendering reference ...')
        timer = Timer()
        timer.start()

        # Create folder for the reference.
        reference_path = Path(outfolder) / 'reference'
        reference_path.mkdir(parents=True, exist_ok=True)

        # Render all materials, MIP levels, UDIMs, and targets.
        for idx, material in enumerate(reference_materials):
            print(f'  Material {idx + 1}/{len(reference_materials)} ...')

            num_mip_levels = config['model']['latents']['num_mip_levels']
            mip_levels = list(range(num_mip_levels)) if self.render_all_mips else [0]
            for mip_level in mip_levels:
                udims = material.udims if self.render_all_udims else [material.udims[0]]
                for udim_id in udims:
                    bitmaps = self._render_reference_material(
                        data_generators,
                        material,
                        mip_level,
                        udim_id,
                        targets,
                    )

                    for target, bitmap in zip(targets, bitmaps, strict=True):
                        name = 'rendering'
                        outname = f'{name}.material{material.id}.mip{mip_level}.udim{udim_id}.{target.name}.{self.format_ext}'
                        if self.format_ext == 'png':
                            bitmap = bitmap.convert(
                                component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True
                            )
                        bitmap.write(reference_path / outname)

        timer.stop()
        print(f'Rendering took {(timer.elapsed()):.2f}s.')

    def render_neural_materials(
        self,
        ckpt: Checkpoint,
        reference_materials: ReferenceMaterials,
        data_generators: DataGenerators,
    ):
        if not self.save_neural:
            return

        timer = Timer()
        timer.start()
        print('Rendering neural scene ...')

        self._prepare_neural_materials(ckpt.model)

        training_targets: list[TrainingTargets] = [TrainingTargets('bsdf')]
        if ckpt.model.aux is not None:
            training_targets += [ckpt.model.aux.targets]

        # Render all materials, MIP levels, UDIMs, and targets.
        for idx, material in enumerate(reference_materials):
            print(f'  Material {idx + 1}/{len(reference_materials)} ...')

            num_mip_levels = ckpt.model.num_mip_levels
            mip_levels = list(range(num_mip_levels)) if self.render_all_mips else [0]
            for mip_level in mip_levels:
                udims = material.udims if self.render_all_udims else [material.udims[0]]
                for udim_id in udims:
                    for targets in training_targets:
                        bitmaps = self._render_neural_material(
                            ckpt.model,
                            data_generators,
                            material,
                            mip_level,
                            udim_id,
                            targets,
                        )

                        for target, bitmap in zip(targets, bitmaps, strict=True):
                            outname = f'rendering.material{material.id}.mip{mip_level}.udim{udim_id}.{target.name}.{self.format_ext}'
                            if self.format_ext == 'png':
                                bitmap = bitmap.convert(
                                    component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True
                                )
                            bitmap.write(ckpt.folder / outname)

        timer.stop()
        print(f'Rendering took {(timer.elapsed()):.2f}s.')

    def _reset_accumulator(self) -> None:
        self._utils_module.accumulator_reset(self._accumulator)

    def _render_with_accumulation(
        self,
        hybrid_pt: HybridPathTracer,
        camera: f2.Camera,
        mip_level: int,
        neural_material: bool,
    ) -> spy.Bitmap:
        """Run one scene's hybrid path tracer for ``self.spp`` samples."""
        self._reset_accumulator()

        for iteration in range(self.spp):
            hybrid_pt.render(camera, self.output_tensor, iteration, mip_level, neural_material)
            self._utils_module.accumulator_update_and_output_inplace(
                self.output_tensor, self._accumulator
            )

        return spy.Bitmap(self.output_tensor.to_numpy())

    def _render_reference_material(
        self,
        data_generators: DataGenerators,
        material: ReferenceMaterial,
        mip_level: int,
        udim_id: int,
        targets: TrainingTargets,
    ) -> list[spy.Bitmap]:
        """Render a reference material using the hybrid path tracer."""
        if targets.name != 'bsdf':
            return self._render_reference_aux_material(
                data_generators,
                material,
                mip_level,
                udim_id,
                targets,
            )

        self._apply_udim_offset(self.reference_scene, udim_id, sign=1.0)
        try:
            if material.falcor2_handle is None:
                raise RuntimeError('Reference material does not have a falcor2 material handle.')
            prepare_scene_material(self.reference_scene, material.falcor2_handle)
            self.reference_hybrid_pt.set_target_material(None, material.texture_resolution)

            bitmap = self._render_with_accumulation(
                self.reference_hybrid_pt,
                self.reference_camera,
                mip_level,
                neural_material=False,
            )
        finally:
            self._apply_udim_offset(self.reference_scene, udim_id, sign=-1.0)

        return [bitmap]

    def _prepare_neural_materials(self, neural_model: NeuralModelCheckpoint) -> None:
        """Set up a neural material for hybrid path tracer rendering."""
        for material in self._neural_materials.values():
            if material.is_valid:
                material.remove()

        self._neural_model = neural_model
        self._neural_materials = {}
        self._model_module = None
        self._hit_kernel_key = None

    def _render_neural_material(
        self,
        neural_model: NeuralModelCheckpoint,
        data_generators: DataGenerators,
        material: ReferenceMaterial,
        mip_level: int,
        udim_id: int,
        targets: TrainingTargets,
    ) -> list[spy.Bitmap]:
        """Render a neural material using the hybrid path tracer."""
        if targets.name != 'bsdf':
            return self._render_neural_aux_material(
                neural_model,
                material,
                mip_level,
                udim_id,
                targets,
            )

        self._apply_udim_offset(self.neural_scene, udim_id, sign=1.0)
        try:
            neural_material = self._get_neural_material(neural_model, material)
            prepare_scene_material(self.neural_scene, neural_material)
            self.neural_hybrid_pt.set_target_material(None, material.texture_resolution)

            bitmap = self._render_with_accumulation(
                self.neural_hybrid_pt,
                self.neural_camera,
                mip_level,
                neural_material=True,
            )
        finally:
            self._apply_udim_offset(self.neural_scene, udim_id, sign=-1.0)

        return [bitmap]

    def _render_neural_aux_material(
        self,
        neural_model: NeuralModelCheckpoint,
        material: ReferenceMaterial,
        mip_level: int,
        udim_id: int,
        targets: TrainingTargets,
    ) -> list[spy.Bitmap]:
        """Render neural aux signals via 2-pass: hit extraction + eval_aux."""
        if neural_model.aux is None:
            raise RuntimeError(
                f'Neural aux target {targets.name!r} requested but the checkpoint has no aux decoder.'
            )

        neural_material = self._get_neural_material(neural_model, material)
        material_id = prepare_scene_material(self.neural_scene, neural_material)

        # Cache the model module used by ``eval_latent_texture()`` and
        # ``eval_aux()``.
        if self._model_module is None:
            self._model_module = spy.Module.load_from_file(
                self.device, 'model/neural_material_model.slang'
            )

        def eval_aux_neural(uv, wi, wo, per_pixel_mip_level, texel_sample):
            assert self._model_module is not None
            latents = self._model_module.eval_latent_texture_bilinear_stochastic(
                neural_model.latent_texture,
                material.id,
                per_pixel_mip_level,
                uv,
                texel_sample,
            )
            return self._model_module.eval_aux(neural_model.aux, latents, wi)

        return self._render_aux_common(
            self.neural_scene,
            self.neural_hybrid_pt,
            self.neural_camera,
            material_id,
            material.texture_resolution,
            neural_model.num_mip_levels,
            mip_level,
            udim_id,
            targets,
            eval_aux_neural,
        )

    def _get_neural_material(
        self,
        neural_model: NeuralModelCheckpoint,
        material: ReferenceMaterial,
    ) -> NeuralMaterial:
        if self._neural_model is not neural_model:
            self._prepare_neural_materials(neural_model)

        neural_material = self._neural_materials.get(material.id)
        if neural_material is None:
            neural_material = self.neural_scene.create_material(NeuralMaterial)
            neural_material.name = f'neural_material_{material.id}'
            neural_material.configure(self.device, neural_model, material.id, self.latent_filter)
            self._neural_materials[material.id] = neural_material
            self._hit_kernel_key = None
        return neural_material

    def _render_reference_aux_material(
        self,
        data_generators: DataGenerators,
        material: ReferenceMaterial,
        mip_level: int,
        udim_id: int,
        targets: TrainingTargets,
    ) -> list[spy.Bitmap]:
        """Render reference aux signals."""
        aux_data_generator = data_generators.aux
        if aux_data_generator is None:
            raise RuntimeError('Auxiliary data generator is not initialized.')

        material_id = prepare_scene_material(self.reference_scene, material.falcor2_handle)

        def eval_aux_reference(uv, wi, wo, per_pixel_mip_level, texel_sample):
            return aux_data_generator.eval_reference(
                material,
                per_pixel_mip_level,
                uv,
                wi,
                wo,
            )

        return self._render_aux_common(
            self.reference_scene,
            self.reference_hybrid_pt,
            self.reference_camera,
            material_id,
            material.texture_resolution,
            mip_level + 1,
            mip_level,
            udim_id,
            targets,
            eval_aux_reference,
        )

    def _render_aux_common(
        self,
        scene: f2.Scene,
        hybrid_pt: HybridPathTracer,
        camera: f2.Camera,
        material_id: int,
        texture_resolution: int,
        num_mip_levels: int,
        fixed_mip_level: int,
        udim_id: int,
        targets: TrainingTargets,
        eval_fn: Callable,
    ) -> list[spy.Bitmap]:
        """Render auxiliary outputs with shared extraction and accumulation.

        Args:
            scene: Native scene containing the material being rendered.
            hybrid_pt: Path tracer whose settings configure hit extraction.
            camera: Camera belonging to ``scene``.
            material_id: falcor2 material system ID of the target material,
                used as TARGET_MATERIAL_ID in the hit extraction kernel.
            eval_fn: Called per SPP iteration with hit and filtering tensors,
                returns a numpy array of shape (H, W, num_channels).
        """
        self._apply_udim_offset(scene, udim_id, sign=1.0)
        try:
            num_channels = targets.num_channels
            render_buffer = spy.Tensor.zeros(
                self.device, shape=(HEIGHT, WIDTH), dtype=f'float[{num_channels}]'
            )
            constants = {
                'MAX_DEPTH': hybrid_pt.max_depth,
                'ENABLE_NEE': hybrid_pt.enable_nee,
                'ENABLE_MIS': hybrid_pt.enable_mis,
                'ENABLE_EMISSIVE_TRIANGLES': hybrid_pt.enable_emissive_triangles,
                'ENABLE_ENV_MAP': hybrid_pt.enable_env_map,
                'TARGET_MATERIAL_ID': material_id,
            }

            # Cache the compiled hit-extraction kernel. Rebuild when the scene's
            # required modules change (e.g. materials added) or when the target
            # ``material_id`` changes.
            required_modules = tuple(scene.requirements.modules)
            key = (id(scene), required_modules, material_id)
            if key != self._hit_kernel_key:
                self._hit_kernel_key = key
                render_module = spy.Module(scene.render_module)
                hit_data_module = spy.Module(
                    self.device.load_module('rendering/hybrid/hit_data_extractor.slang'),
                    link=[render_module] + [spy.Module(m) for m in scene.requirements.modules],
                )
                self._hit_kernel_base = hit_data_module.extract_hit_data.constants(
                    constants
                ).type_conformances(scene.requirements.type_conformances)

            # Rebind scene state every call; UDIM offsets and other scene edits
            # can mutate uniforms between auxiliary targets. Detecting those
            # changes is not worth the bookkeeping for checkpoint renderings.
            assert self._hit_kernel_base is not None
            hit_kernel = self._hit_kernel_base.write(scene.bind)
            self._hit_kernel = hit_kernel

            uv = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='float2')
            wi = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='float3')
            wo = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='float3')
            hit_mask = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='uint')
            mip_level = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='uint')
            texel_sample = spy.Tensor.empty(self.device, (HEIGHT, WIDTH), dtype='float')

            for iteration in range(self.spp):
                hit_kernel.call(
                    camera,
                    uv,
                    wi,
                    wo,
                    hit_mask,
                    mip_level,
                    texel_sample,
                    iteration,
                    fixed_mip_level,
                    texture_resolution,
                    num_mip_levels,
                    tid=spy.grid((HEIGHT, WIDTH)),
                )
                L = eval_fn(uv, wi, wo, mip_level, texel_sample)
                self._aux_accumulator_module.accumulate_aux_masked(
                    L, hit_mask, result=render_buffer
                )

            render_buffer_np = render_buffer.to_numpy().astype(np.float32) / self.spp
            return split_aux_channels(render_buffer_np, targets)
        finally:
            self._apply_udim_offset(scene, udim_id, sign=-1.0)

    def _apply_udim_offset(self, scene: f2.Scene, udim_id: int, sign: float) -> None:
        offset = udim_offset(udim_id)
        apply_udim_offset_to_scene(scene, (sign * offset[0], sign * offset[1]))
        scene.update()
