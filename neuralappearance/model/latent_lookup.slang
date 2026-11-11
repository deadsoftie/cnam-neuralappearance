// SPDX-License-Identifier: Apache-2.0

// Stochastic latent lookups of any latent texture, dense or NTC, with one
// kernel: the same code path times and tests every latent texture type.

__exported import model.base_types;
__exported import model.latent_texture;
__exported import model.ntc_texture;

void latent_lookup<let NumLatents : int, LatentTexture : ILatentTexture<NumLatents>>(
    LatentTexture latents,
    int material_id,
    int mip_level,
    float2 uv,
    float sample,
    out float4 a,
    out float4 b)
{
    let values = latents.get_bilinear_stochastic({ material_id, mip_level, uv }, sample);
    a = float4(values[0], values[1], values[2], values[3]);
    b = float4(values[4], values[5], values[6], values[7]);
}
