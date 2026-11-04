# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import falcor2 as f2
import slangpy as spy
from slangpy.core.native import unpack_arg

if TYPE_CHECKING:
    from model import NeuralModelCheckpoint

LATENT_FILTERS = ('stochastic', 'bilinear')


class NeuralMaterial(f2.Material):
    """Expose a trained neural appearance model as a falcor2 material.

    The model architecture becomes generated Slang type aliases, while model
    parameters and textures live in a bound configuration buffer. A source hash
    gives each distinct architecture its own material type.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module: spy.SlangModule | None = None
        self.spy_module: spy.Module | None = None
        self.config_tensor: spy.Tensor | None = None
        self.module_hash = ''
        self.reference_material_id = 0

    def configure(
        self,
        device: spy.Device,
        neural_model: NeuralModelCheckpoint,
        material_id: int,
        latent_filter: str = 'stochastic',
    ) -> None:
        """Compile and bind a neural material for a reference material ID.

        ``latent_filter`` selects the latent lookup per shading point:
        ``stochastic`` (one bilinear neighbor) or ``bilinear`` (blend of 4).
        """
        self.reference_material_id = int(material_id)

        generated_source = self.code_generate(neural_model, latent_filter)
        source_hash = hashlib.sha256(generated_source.encode('utf-8')).hexdigest()[:16]

        module_changed = source_hash != self.module_hash
        if module_changed:
            self.module_hash = source_hash
            source_path = Path(__file__).parent / 'neural_material.slang'
            source = source_path.read_text()
            source += generated_source
            source = source.replace('<HASH>', self.module_hash)

            self.module = device.load_module_from_source(
                f'neural_material_{self.module_hash}',
                source,
            )
            self.spy_module = spy.Module(self.module)
            self.slang_type_name = f'NeuralMaterial_{self.module_hash}'

        self._write_config_buffer(device, neural_model)

        dirty_flags = f2.Material.DirtyFlags.properties
        if module_changed:
            dirty_flags |= f2.Material.DirtyFlags.resources
        self.mark_dirty(dirty_flags)

    def required_module(self) -> spy.SlangModule | None:
        return self.module

    def write_to_cursor(self, cursor: spy.ShaderCursor) -> None:
        if self.config_tensor is None:
            raise RuntimeError('NeuralMaterial.configure() must be called before scene.update().')

        cursor['material_id'] = self.reference_material_id
        cursor['config_handle'] = f2.to_handle(self.config_tensor.storage).data

    def code_generate(
        self, neural_model: NeuralModelCheckpoint, latent_filter: str = 'stochastic'
    ) -> str:
        """Generate concrete Slang aliases for the checkpoint's model types."""

        if latent_filter not in LATENT_FILTERS:
            raise ValueError(f'latent_filter must be one of {LATENT_FILTERS}, got {latent_filter}')
        latents = neural_model.latent_texture
        decoder = neural_model.decoder
        if latents.num_mip_levels != neural_model.num_mip_levels:
            raise ValueError(
                f'latent texture has {latents.num_mip_levels} mip levels, '
                f'the model {neural_model.num_mip_levels}'
            )

        if neural_model.sampler is not None:
            sampler_type = neural_model.sampler.type_name
        else:
            sampler_type = f'CosineHemisphereSampler<{latents.num_channels}>'

        if neural_model.aux is not None:
            aux_type = neural_model.aux.type_name
        else:
            aux_type = f'DummyAuxDecoder<{latents.num_channels}, 1>'

        slang_module = getattr(latents, 'slang_module', None)
        return _NEURAL_MATERIAL_CONFIG_TEMPLATE.format(
            latents_import=f'import {slang_module};' if slang_module else '',
            latents_type=latents.type_name,
            decoder_type=decoder.type_name,
            sampler_type=sampler_type,
            aux_type=aux_type,
            num_mip_levels=neural_model.num_mip_levels,
            num_latents=latents.num_channels,
            bilinear_latents='true' if latent_filter == 'bilinear' else 'false',
        )

    def _write_config_buffer(
        self,
        device: spy.Device,
        neural_model: NeuralModelCheckpoint,
    ) -> None:
        if self.spy_module is None:
            raise RuntimeError('Neural material module has not been compiled.')

        config_type_name = f'NeuralMaterialConfig_{self.module_hash}'
        config_type = self.spy_module[config_type_name]
        self.config_tensor = spy.Tensor.empty(device, shape=(1,), dtype=config_type)

        config: dict[str, Any] = {}
        for name, component in neural_model.components():
            if component is not None:
                config[name] = unpack_arg(component)

        cursor = self.config_tensor.cursor()
        cursor[0].write(config)
        cursor.apply()


_NEURAL_MATERIAL_CONFIG_TEMPLATE = """
{latents_import}
public typealias LatentTexture_<HASH> = {latents_type};
public typealias Decoder_<HASH> = {decoder_type};
public typealias Sampler_<HASH> = {sampler_type};
public typealias AuxDecoder_<HASH> = {aux_type};

public struct NeuralMaterialConfig_<HASH>
{{
    static const int num_mip_levels = {num_mip_levels};
    static const int num_latents = {num_latents};
    static const bool bilinear_latents = {bilinear_latents};
    public LatentTexture_<HASH> latents;
    public Decoder_<HASH> decoder;
    public Sampler_<HASH> sampler;
    public AuxDecoder_<HASH> aux;
}};
"""
