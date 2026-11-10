// SPDX-License-Identifier: Apache-2.0

// E2E loss for the NTC trainer: RTNAM BSDF samples at the SDK's sampled texel
// centers and the frozen BSDF decoder on the decoded latent. The gradient of the
// loss with respect to the linear latent reaches the network outputs as the
// SDK's L2 gradient does, without the channel mapping scale.
import slangpy;
__exported import ntc_training.ntc_train;
__exported import neural_networks;
__exported import datagen.data_generator;
__exported import model.neural_material_model;

// One BSDF sample per batch texel: color mip and texel center from the SDK
// sampling, directions as the dense data generator. ``wo`` stays zero when the
// strategy leaves it to importance sampling (``fill_importance_sampled_wo``).
void ntc_bsdf_samples(
    NtcNetwork net,
    NtcTrainState st,
    DataGenerator::Config config,
    int material_id,
    int material_backend_id,
    int2 call_id,
    inout UniformSampleGenerator sg,
    out TrainingSample<3, 1> sample)
{
    let texel = st.texels[call_id.x];
    let color_mip = net.color_mips[texel.x];
    sample.material_id = material_id;
    sample.material_backend_id = material_backend_id;
    sample.mip_level = texel.x;
    sample.uv = (float2(texel.y, texel.z) + 0.5) / float2(color_mip.x, color_mip.y);
    sample.target = {};
    sample.encoder_inputs = {};
    bool sampled_wo;
    sample.valid = DataGenerator::sample_directions(config, sg, sample.wi, sample.wo, sampled_wo);
    if (!sampled_wo)
        sample.wo = float3(0.f);
}

[BackwardDifferentiable]
float bsdf_loss<BsdfDecoder : IBsdfDecoder<NtcLatents>, LossFunction : ILossFunction<3>>(
    no_diff BsdfDecoder decoder,
    float latents[NtcLatents],
    no_diff TrainingSample<3, 1> sample,
    no_diff LossFunction loss_function)
{
    float prediction[3] = decoder.forward({ sample.material_id, sample.mip_level, latents, sample.wi, sample.wo });
    return loss_function.eval(prediction, sample.target) * (sample.valid ? 1.f : 0.f);
}

// Forward, 7.1 loss through the frozen decoder and backward for batch texel ``call_id.x``.
// ``control_variate``: subtract the gradient at the dense latents in ``st.reference``.
void ntc_train_texel_bsdf<BsdfDecoder : IBsdfDecoder<NtcLatents>, LossFunction : ILossFunction<3>>(
    NtcNetwork net,
    NtcTrainState st,
    NtcLossParams lp,
    BsdfDecoder decoder,
    TrainingSample<3, 1> sample,
    LossFunction loss_function,
    int control_variate,
    int2 call_id)
{
    NeuralConfig::set_batch_index(0);
    let t = call_id.x;
    let f = forward_texel(net, st, t);

    float latents[NtcLatents];
    for (int c = 0; c < NtcLatents; ++c)
        latents[c] = linear_latent(net, float(f.y3[c]), c);
    st.texel_loss[t] = bsdf_loss(decoder, latents, sample, loss_function);

    var pair = diffPair(latents);
    bwd_diff(bsdf_loss<BsdfDecoder, LossFunction>)(decoder, pair, sample, loss_function, 1.f);

    float g[NtcOutputs] = {};
    for (int c = 0; c < NtcLatents; ++c)
        g[c] = pair.d[c] * lp.gradient_scale;

    // Control variate: the dense latents minimize this loss, so their gradient on the
    // same sample has expectation zero and carries most of the direction noise.
    if (control_variate != 0)
    {
        let texel = st.texels[t];
        let ref = st.reference_offsets[texel.x] + (texel.z * net.color_mips[texel.x].x + texel.y) * NtcLatents;
        float dense[NtcLatents];
        for (int c = 0; c < NtcLatents; ++c)
            dense[c] = st.reference[ref + c];
        var dense_pair = diffPair(dense);
        bwd_diff(bsdf_loss<BsdfDecoder, LossFunction>)(decoder, dense_pair, sample, loss_function, 1.f);
        for (int c = 0; c < NtcLatents; ++c)
            g[c] -= dense_pair.d[c] * lp.gradient_scale;
    }
    backward_texel(net, st, lp, t, f, g);
}

// Loss of the frozen decoder at given latents, for gradient checks.
float ntc_bsdf_loss_at<BsdfDecoder : IBsdfDecoder<NtcLatents>, LossFunction : ILossFunction<3>>(
    BsdfDecoder decoder,
    float latents[NtcLatents],
    TrainingSample<3, 1> sample,
    LossFunction loss_function)
{
    return bsdf_loss(decoder, latents, sample, loss_function);
}

// Gradient of that loss with respect to the latents, for gradient checks.
float[NtcLatents] ntc_bsdf_loss_gradient<BsdfDecoder : IBsdfDecoder<NtcLatents>, LossFunction : ILossFunction<3>>(
    BsdfDecoder decoder,
    float latents[NtcLatents],
    TrainingSample<3, 1> sample,
    LossFunction loss_function)
{
    NeuralConfig::set_batch_index(0);
    var pair = diffPair(latents);
    bwd_diff(bsdf_loss<BsdfDecoder, LossFunction>)(decoder, pair, sample, loss_function, 1.f);
    return pair.d;
}

void ntc_bsdf_sample_fields(TrainingSample<3, 1> sample, out float4 uv_mip_valid, out float3 wi, out float3 wo,
                            out float3 target)
{
    uv_mip_valid = float4(sample.uv, float(sample.mip_level), sample.valid ? 1.f : 0.f);
    wi = sample.wi;
    wo = sample.wo;
    target = float3(sample.target[0], sample.target[1], sample.target[2]);
}
