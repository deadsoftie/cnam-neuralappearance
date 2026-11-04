// SPDX-License-Identifier: Apache-2.0

// clang-format off
// The NTC SDK shader headers are found through the compiler include paths of
// the device; ``ntc_texture.py`` documents the expected directory.
#include "libntc/shaders/InferenceCoopVec.hlsli"
import slangpy;
import model.base_types;
import util.texture_helpers;

static const int NtcNumLatents = 8;

// One NTC texture set resident on the GPU: packed BGRA4 latents, the SDK
// latent sampler, the FP8 weight set in CoopVec layout and the SDK constants.
// As a latent texture it holds one material without UDIMs: ``material_id`` is
// ignored, ``mip_level`` is the color mip and UVs wrap. Every lookup decodes at
// integer texels only; the lookups pick texels as the dense ``Texture`` does.
// Decoded latents carry no gradients.
struct NtcTexture : ILatentTexture<NtcNumLatents>
{
    Texture2DArray.Handle latents;
    SamplerState.Handle sampler;
    ByteAddressBuffer.Handle weights;
    NtcTextureSetConstants constants;

    int get_texture_resolution(int material_id)
    {
        return constants.imageWidth;
    }

    [BackwardDifferentiable]
    float[NtcNumLatents] get_nearest(LatentTextureInput input)
    {
        return no_diff decode_nearest(input.mip_level, input.uv);
    }

    // Decodes the 4 bilinear neighbors and blends the decoded values.
    [BackwardDifferentiable]
    float[NtcNumLatents] get_bilinear(LatentTextureInput input)
    {
        return no_diff decode_bilinear(input.mip_level, input.uv);
    }

    [BackwardDifferentiable]
    float[NtcNumLatents] get_bilinear_stochastic(LatentTextureInput input, float sample)
    {
        let size = uint2(mip_size(input.mip_level));
        let uv = sample_bilinear_texel_center(input.uv, size, true, sample);
        return no_diff decode_nearest(input.mip_level, uv);
    }

    float[NtcNumLatents] decode_nearest(int mip_level, float2 uv)
    {
        let size = uint2(mip_size(mip_level));
        let texel = int2(round(uv_to_fractional_texel(uv, size, true)));
        return decode_latents(texel, mip_level);
    }

    float[NtcNumLatents] decode_bilinear(int mip_level, float2 uv)
    {
        uint2 texels[4];
        float weights[4];
        get_bilinear_interpolation_texels_and_weights(
            uv, uint2(mip_size(mip_level)), true, texels, weights);

        float[NtcNumLatents] result = {};
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
        {
            let code = decode_latents(int2(texels[i]), mip_level);
            [ForceUnroll]
            for (int l = 0; l < NtcNumLatents; ++l)
                result[l] += weights[i] * code[l];
        }
        return result;
    }

    // Wraps ``texel`` into color mip ``mip_level`` and decodes it. Zeros for a
    // mip level the texture set does not have.
    float[NtcNumLatents] decode_latents(int2 texel, int mip_level)
    {
        float outputs[NTC_MLP_OUTPUT_CHANNELS];
        decode(wrap_texel(texel, mip_level), mip_level, outputs);

        float[NtcNumLatents] result;
        [ForceUnroll]
        for (int i = 0; i < NtcNumLatents; ++i)
            result[i] = outputs[i];
        return result;
    }

    // Decodes the integer texel of color mip ``mip_level``. The texel must lie
    // inside the mip: the SDK clamps out-of-range texels. Returns false for a
    // mip level the texture set does not have.
    // Same steps as the SDK's ``DecompressCoopVecFP8.slang``. SDK functions that
    // take ``NtcTextureSetConstants`` by value crash inside the slangpy call,
    // so the constants are passed to the SDK field by field.
    bool decode(int2 texel, int mip_level, out float outputs[NTC_MLP_OUTPUT_CHANNELS])
    {
        [unroll]
        for (int ch = 0; ch < NTC_MLP_OUTPUT_CHANNELS; ++ch)
            outputs[ch] = 0;

        if (mip_level < 0 || mip_level >= constants.imageMips)
            return false;

        let uv = (float2(texel) + 0.5) / mip_size(mip_level);
        let color_mip = NtcUnpackColorMipConstants(constants.colorMips[mip_level]);

        CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> inputs;
        if (!NtcPrepareNetworkInputsInternal_FP16(latents, sampler, texel, uv, color_mip, inputs))
            return false;

        CoopVec<float, NTC_MLP_OUTPUT_CHANNELS> network_outputs;
        NtcEvaluateMLP_CoopVec_FP8(weights, 0, constants.networkWeightOffsets,
            constants.networkBiasOffsets, constants.networkScaleOffsets, inputs, network_outputs);

        [unroll]
        for (int ch = 0; ch < NTC_MLP_OUTPUT_CHANNELS; ++ch)
        {
            let stored_space = int(constants.channelColorSpaces >> (ch * 2)) & 3;
            outputs[ch] = NtcConvertColorSpace(
                network_outputs[ch], stored_space, NtcColorSpace_Linear);
        }
        return true;
    }

    int2 wrap_texel(int2 texel, int mip_level)
    {
        let size = mip_size(mip_level);
        return ((texel % size) + size) % size;
    }

    int2 mip_size(int mip_level)
    {
        return int2(
            max(constants.imageWidth >> mip_level, 1), max(constants.imageHeight >> mip_level, 1));
    }
}

// Decodes every texel of one color mip; ``texel`` comes from ``spy.grid``.
// ``offset`` shifts the decoded texel, wrapped into the mip.
void ntc_decode_mip(
    NtcTexture ntc, int mip_level, int2 offset, int2 texel, out float4 a, out float4 b)
{
    let outputs = ntc.decode_latents(texel + offset, mip_level);
    a = float4(outputs[0], outputs[1], outputs[2], outputs[3]);
    b = float4(outputs[4], outputs[5], outputs[6], outputs[7]);
}

// Runs one latent lookup of ``ntc`` per call; ``filter`` 0 nearest,
// 1 bilinear, 2 stochastic.
void ntc_lookup(
    NtcTexture ntc, int filter, int mip_level, float2 uv, float sample, out float4 a, out float4 b)
{
    LatentTextureInput input = { 0, mip_level, uv };
    float[NtcNumLatents] outputs;
    if (filter == 0)
        outputs = ntc.get_nearest(input);
    else if (filter == 1)
        outputs = ntc.get_bilinear(input);
    else
        outputs = ntc.get_bilinear_stochastic(input, sample);
    a = float4(outputs[0], outputs[1], outputs[2], outputs[3]);
    b = float4(outputs[4], outputs[5], outputs[6], outputs[7]);
}
