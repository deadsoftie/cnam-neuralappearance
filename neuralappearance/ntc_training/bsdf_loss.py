# SPDX-License-Identifier: Apache-2.0

"""E2E loss for the NTC trainer (``ntc_bsdf.slang``).

Per step and batch texel: directions from the dense data generator's strategy,
importance-sampled ``wo`` where the strategy asks for it, the reference BSDF
prefiltered for the texel's color mip (``fill_target_bsdf_lod``), all with the
dense run's data generation settings and a sample generator seeded with the
fine-tune seed. At mip 0 the prefilter width is zero and the target is one
evaluation (``ntc_bsdf_target.slang``) unless ``prefilter_mip0``; ``prefilter_samples``
replaces the dense run's prefilter sample count. The frozen BSDF
decoder of the dense checkpoint evaluates the decoded latent; the dense loss
function compares.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

MODULE = Path(__file__).with_name('ntc_bsdf.slang')
TARGET_MODULE = 'ntc_training/ntc_bsdf_target.slang'


class BsdfLoss:
    loss_divisor = 1

    def __init__(self, device: spy.Device, config: dict, reference_materials: Any, decoder: Any, seed: int,
                 batch: int, prefilter_mip0: bool = False, prefilter_samples: int | None = None):
        from datagen import FalcorBsdfDataGenerator
        from training import LossFunction, TrainingTargets

        config = copy.deepcopy(config)
        config['data_generation']['seed'] = seed
        config['data_generation']['color_augmentation']['enabled'] = False
        (self.material,) = reference_materials
        if list(self.material.udims) != [1001]:
            raise ValueError(f'one material tile 1001 expected, found UDIMs {self.material.udims}')
        self.generator = FalcorBsdfDataGenerator(device, config, reference_materials, 'BsdfDirectOptimization')
        self.generator.sample_generator.reshape((1, batch))
        self.datagen_config = self.generator.datagen_config.copy()
        if prefilter_samples is not None:
            self.datagen_config['num_prefilter_samples'] = prefilter_samples
        self.samples = spy.Tensor.empty(device, shape=(1, batch), dtype=self.generator.training_sample_type)
        self.texture_resolution = spy.Tensor.from_numpy(
            device, np.array([self.material.texture_resolution], np.int32))
        self.loss_function = LossFunction(config['training']['loss_function'], TrainingTargets('bsdf'))
        self.decoder = decoder
        self.device = device
        self.module = spy.Module.load_from_file(device, str(MODULE))
        self.target_kernel = self.generator.fill_target_kernel
        if not prefilter_mip0:
            self.target_kernel = (
                reference_materials.load_scene_module(device, TARGET_MODULE)['ntc_bsdf_target'].as_func()
                .type_conformances(reference_materials.conformances)
                .write(reference_materials.bind_scene)
                .map(texture_resolution=(0,), sg=(0, 1), sample=(0, 1)))

    def run(self, kernel, *args, **kwargs) -> None:
        kernel(*args, **kwargs)
        self.device.wait()

    def prepare(self, network: dict, state: dict, dispatch=None) -> None:
        """Fill the BSDF samples for the texels of the current batch. ``dispatch``: the trainer's, else each
        kernel runs and the device is waited for."""
        dispatch = dispatch or self.run
        sg = self.generator.sample_generator
        dispatch(self.module.ntc_bsdf_samples, network, state, self.datagen_config, self.material.id,
                 self.material.backend_id, spy.call_id(), sg, self.samples)
        dispatch(self.generator.fill_importance_sampled_wo_kernel, config=self.datagen_config,
                 texture_resolution=self.texture_resolution, sg=sg, sample=self.samples)
        dispatch(self.target_kernel, config=self.datagen_config, texture_resolution=self.texture_resolution, sg=sg,
                 sample=self.samples)

    def train(self, network: dict, state: dict, loss_params: dict, control_variate: bool, dispatch=None) -> None:
        """``control_variate``: subtract the gradient at the dense latents in the trainer's reference."""
        (dispatch or self.run)(self.module.ntc_train_texel_bsdf, network, state, loss_params, self.decoder,
                               self.samples, self.loss_function, int(control_variate), spy.call_id())
