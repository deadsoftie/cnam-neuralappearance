// SPDX-License-Identifier: Apache-2.0

// clang-format off
// The NTC SDK shader headers are found through the compiler include paths of
// the device; ``ntc_texture.py`` documents the expected directory.
#include "libntc/shaders/InferenceCoopVec.hlsli"
import slangpy;

// One NTC texture set resident on the GPU: packed BGRA4 latents, the SDK
// latent sampler, the FP8 weight set in CoopVec layout and the SDK constants.
struct NtcTexture
{
    Texture2DArray.Handle latents;
    SamplerState.Handle sampler;
    ByteAddressBuffer.Handle weights;
    NtcTextureSetConstants constants;

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

    int2 mip_size(int mip_level)
    {
        return int2(
            max(constants.imageWidth >> mip_level, 1), max(constants.imageHeight >> mip_level, 1));
    }
}

// Wraps an integer texel into color mip ``mip_level``.
int2 ntc_wrap_texel(NtcTexture ntc, int2 texel, int mip_level)
{
    let size = ntc.mip_size(mip_level);
    return ((texel % size) + size) % size;
}

// Decodes every texel of one color mip; ``texel`` comes from ``spy.grid``.
// ``offset`` shifts the decoded texel, wrapped into the mip.
void ntc_decode_mip(
    NtcTexture ntc, int mip_level, int2 offset, int2 texel, out float4 a, out float4 b)
{
    float outputs[NTC_MLP_OUTPUT_CHANNELS];
    ntc.decode(ntc_wrap_texel(ntc, texel + offset, mip_level), mip_level, outputs);
    a = float4(outputs[0], outputs[1], outputs[2], outputs[3]);
    b = float4(outputs[4], outputs[5], outputs[6], outputs[7]);
}
