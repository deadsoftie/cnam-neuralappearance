# SPDX-License-Identifier: Apache-2.0

"""GPU resources for decoding an NTC texture set with the SDK shader code.

The input is a parsed ``.ntc`` file: an object with ``width``, ``height``,
``num_color_mips``, ``num_channels``, ``channel_color_spaces``, ``latents``
(images with packed BGRA4 ``layers``), ``color_mips`` and ``mlp(first_weight_type)``
(layers with raw ``weights``, ``biases`` and ``scales`` bytes). Shaders that import
``model.ntc_texture`` need the directory containing ``libntc/shaders`` (the
NTC SDK ``include`` directory) in the device's compiler include paths.

``NtcTexture`` takes the place of a checkpoint's ``LatentTexture`` for one
material: it implements the Slang ``ILatentTexture`` interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import slangpy as spy

MAX_MIPS = 16
MLP_LAYERS = 4
LAYER_SHAPES = ((48, 64), (64, 48), (48, 32), (32, 16))
FP8_LAYER_TYPES = ('FloatE4M3', 'FloatE4M3', 'FloatE4M3', 'Int8')
COLOR_SPACE_CODES = {'Linear': 0, 'sRGB': 1, 'HLG': 2}
NUM_LATENTS = 8


class NtcTexture:
    type_name = 'NtcTexture'
    slang_module = 'model.ntc_texture'
    num_channels = NUM_LATENTS

    def __init__(self, device: spy.Device, ntc: Any):
        if ntc.num_channels != NUM_LATENTS:
            raise ValueError(f'expected {NUM_LATENTS} channels, the file has {ntc.num_channels}')
        if ntc.num_color_mips > MAX_MIPS:
            raise ValueError(f'at most {MAX_MIPS} color mips, the file has {ntc.num_color_mips}')
        self.device = device
        self.num_layers = len(ntc.latents[0].layers)
        self.num_mip_levels = ntc.num_color_mips

        self.latents = self._create_latents(ntc)
        # Matches the latent sampler of the SDK decompression pass: bilinear,
        # wrap, no mip filtering.
        self.sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap,
            max_anisotropy=1,
            mip_lod_bias=0.0,
        )
        self.weights, offsets = self._create_weights(ntc)
        self.constants = self._constants(ntc, *offsets)

    def get_this(self) -> dict:
        return {
            'latents': self.latents.descriptor_handle_ro,
            'sampler': self.sampler.descriptor_handle,
            'weights': self.weights.descriptor_handle_ro,
            'constants': self.constants,
        }

    def _create_latents(self, ntc: Any) -> spy.Texture:
        top = ntc.latents[0]
        texture = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.bgra4_unorm,
            width=top.width,
            height=top.height,
            array_length=self.num_layers,
            mip_count=len(ntc.latents),
            usage=spy.TextureUsage.shader_resource,
        )
        for mip, image in enumerate(ntc.latents):
            for layer, data in enumerate(image.layers):
                texture.copy_from_numpy(np.array(data, dtype=np.uint16), layer=layer, mip=mip)
        return texture

    def _create_weights(
        self, ntc: Any
    ) -> tuple[spy.Buffer, tuple[list[int], list[int], list[int]]]:
        mlp = ntc.mlp('FloatE4M3')
        shapes = tuple((layer.input_channels, layer.output_channels) for layer in mlp.layers)
        types = tuple(layer.weight_type for layer in mlp.layers)
        if shapes != LAYER_SHAPES or types != FP8_LAYER_TYPES:
            raise ValueError(f'unexpected FP8 weight set: {shapes}, {types}')
        if mlp.weight_layout != 'RowMajor':
            raise ValueError(f'unexpected weight layout {mlp.weight_layout}')

        element_types = [spy.DataType.float8_e4m3] * 3 + [spy.DataType.int8]
        sizes = [
            self.device.get_coop_vec_matrix_size(
                n_out, n_in, spy.CoopVecMatrixLayout.inferencing_optimal, element_type
            )
            for (n_in, n_out), element_type in zip(LAYER_SHAPES, element_types, strict=True)
        ]
        weight_offsets = [sum(sizes[:i]) for i in range(MLP_LAYERS)]

        # Bias vectors of layers 0 to 2, then scale and bias of layer 3, after
        # the converted matrices (``ntc/src/WeightLayout.cpp``).
        tail = bytearray()
        bias_offsets, scale_offsets = [], []
        base = sum(sizes)
        for layer in mlp.layers[:-1]:
            bias_offsets.append(base + len(tail))
            scale_offsets.append(0)
            tail += layer.biases
        last = mlp.layers[-1]
        scale_offsets.append(base + len(tail))
        tail += last.scales
        bias_offsets.append(base + len(tail))
        tail += last.biases

        data = np.zeros(base + len(tail), dtype=np.uint8)
        data[base:] = np.frombuffer(bytes(tail), dtype=np.uint8)
        weights = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=data,
        )

        row_major = b''.join(layer.weights for layer in mlp.layers)
        source = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            data=np.array(bytearray(row_major), dtype=np.uint8),
        )
        encoder = self.device.create_command_encoder()
        src_offset = 0
        for (n_in, n_out), element_type, dst_offset, layer in zip(
            LAYER_SHAPES, element_types, weight_offsets, mlp.layers, strict=True
        ):
            # Built by hand: the device query rejects a row-major FP8 layout.
            src_desc = spy.CoopVecMatrixDesc({
                'rows': n_out,
                'cols': n_in,
                'layout': spy.CoopVecMatrixLayout.row_major,
                'element_type': element_type,
                'size': len(layer.weights),
                'offset': src_offset,
                'row_col_stride': n_in,
            })
            dst_desc = self.device.create_coop_vec_matrix_desc(
                n_out, n_in, spy.CoopVecMatrixLayout.inferencing_optimal, element_type, dst_offset
            )
            encoder.convert_coop_vec_matrix(weights, dst_desc, source, src_desc)
            src_offset += len(layer.weights)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return weights, (weight_offsets, bias_offsets, scale_offsets)

    def _constants(
        self,
        ntc: Any,
        weight_offsets: list[int],
        bias_offsets: list[int],
        scale_offsets: list[int],
    ) -> dict:
        """Return the fields of ``NtcTextureSetConstants`` (``InferenceConstants.h``)."""
        mips = [spy.int4(0, 0, 0, 0)] * MAX_MIPS
        for level, m in enumerate(ntc.color_mips):
            mips[level] = spy.int4(
                m.latent_mip, _float_bits(m.position_lod), _float_bits(m.position_scale), 0
            )
        color_spaces = 0
        for channel, name in enumerate(ntc.channel_color_spaces):
            color_spaces |= COLOR_SPACE_CODES[name] << (2 * channel)
        return {
            'colorMips': mips,
            'networkWeightOffsets': spy.int4(*weight_offsets),
            'networkBiasOffsets': spy.int4(*bias_offsets),
            'networkScaleOffsets': spy.int4(*scale_offsets),
            'imageWidth': ntc.width,
            'imageHeight': ntc.height,
            'imageMips': ntc.num_color_mips,
            'pad0': 0,
            'validChannelMask': (1 << ntc.num_channels) - 1,
            'channelColorSpaces': color_spaces,
            'pad1': 0,
            'pad2': 0,
        }


def _float_bits(value: float) -> int:
    return int(np.array(value, dtype=np.float32).view(np.int32))
