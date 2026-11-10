// SPDX-License-Identifier: Apache-2.0

// One training step of the NTC SDK compression (``TextureSet.cpp``,
// ``RegressionKernels.h``, ``Optimizer.cu``, ``Quantizer.cu``) as separate
// kernels. The network runs on cooperative vectors (tensor cores) in FP16 as the
// SDK's ``tin`` MLP: inputs, encoding, activations and quantization as
// ``ntc_forward.slang``, matrix products in hardware, backward in FP16 with
// straight-through quantization. Each warp accumulates its weight gradients into
// its own slot; slots are summed in a fixed order. Latent gradients are summed as
// 64-bit fixed point (``LatentGradScale``). Every reduction is independent of
// thread order.
import slangpy;
__exported import ntc_training.ntc_forward;

static const int NtcParams = 8352;
static const int NtcBiasStart = 8192;
static const int NtcLayerWeightOffsets[4] = { 0, 3072, 6144, 7680 };
static const int NtcLayerBiasOffsets[4] = { 8192, 8256, 8304, 8336 };
static const float LatentGradScale = 1099511627776.0; // 2^40
static const float LatentStep = 1.0 / 15.0;

// ``HashBasedRNG`` (``CudaUtils.h``).
struct HashRng
{
    uint seed;
    uint index;

    __init(uint linear_index, uint offset)
    {
        var a = linear_index;
        a = (a + 0x7ed55d16u) + (a << 12);
        a = (a ^ 0xc761c23cu) ^ (a >> 19);
        a = (a + 0x165667b1u) + (a << 5);
        a = (a + 0xd3a2646cu) ^ (a << 9);
        a = (a + 0xfd7046c5u) + (a << 3);
        a = (a ^ 0xb55a4f09u) ^ (a >> 16);
        seed = a + offset;
        index = 1;
    }

    [mutating]
    uint next_uint()
    {
        var k = index * 0xcc9e2d51u;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593u;
        var h = seed ^ k;
        h = ((h << 13) | (h >> 19)) * 5u + 0xe6546b64u;
        h ^= 4u;
        h ^= h >> 16;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        h *= 0xc2b2ae35u;
        h ^= h >> 16;
        index++;
        return h;
    }

    [mutating]
    float next_float()
    {
        return asfloat((next_uint() & 0x7fffffu) | 0x3f800000u) - 1.0;
    }
}

struct NtcSampling
{
    RWStructuredBuffer<int4> texels;    // color mip, x, y
    float cdf[16];
    int num_mips;
    int width;
    int height;
    uint seed;
}

// Texel ``t`` of the batch: warp ``t / 32`` picks a color mip and a run of 32
// texels in a row (``RegressionKernel``).
void ntc_sample(NtcSampling s, int t)
{
    let warp = uint(t) / 32;
    let lane = t % 32;
    var rng = HashRng(warp, s.seed);
    let r = rng.next_float();
    int mip = 0;
    for (int i = s.num_mips - 1; i >= 0; --i)
        if (r < s.cdf[i])
            mip = i;
    let w = max(s.width >> mip, 1);
    let h = max(s.height >> mip, 1);
    let x_offset = rng.next_float();
    let y_offset = rng.next_float();
    int x = int(floor(x_offset * float(w))) + lane;
    int y = int(floor(y_offset * float(h)));
    y += x / w;
    x = x % w;
    y = y % h;
    s.texels[t] = int4(mip, x, y, 0);
}

struct NtcTrainState
{
    RWStructuredBuffer<float> latents_base;
    RWStructuredBuffer<float> latents_q;
    RWStructuredBuffer<float> latent_m1;
    RWStructuredBuffer<float> latent_m2;
    RWStructuredBuffer<int64_t> latent_grad;
    RWStructuredBuffer<uint> grad_mask;
    RWStructuredBuffer<float> weights_base;
    RWStructuredBuffer<float> weights_q;
    RWStructuredBuffer<float> weight_m1;
    RWStructuredBuffer<float> weight_m2;
    RWStructuredBuffer<float> weight_grad;
    RWStructuredBuffer<int4> texels;
    RWStructuredBuffer<float> output_grads; // per texel, the gradient at the network outputs
    RWStructuredBuffer<float> texel_loss;
    RWStructuredBuffer<float> partials;
    StructuredBuffer<float> reference;  // per color mip, 8 channels per texel
    ByteAddressBuffer coop_weights;     // FP16 matrices in the training-optimal layout
    ByteAddressBuffer coop_biases;      // FP16 bias vectors
    RWByteAddressBuffer grad_slots;     // per warp: weight gradients as ``coop_weights``, then biases
    StructuredBuffer<int> slot_index;   // per parameter, its FP16 index in a slot
    int4 coop_weight_offsets;           // bytes, in ``coop_weights`` and in a slot
    int4 coop_bias_offsets;             // bytes, in ``coop_biases`` and after ``slot_bias_start``
    int slot_bytes;
    int slot_bias_start;
    int reference_offsets[16];
    int pixel_offsets[8];               // first pixel of each neural mip, SDK order
    int num_neural_mips;
    int total_pixels;
    int batch;
    int chunk;
}

struct NtcLossParams
{
    float gradient_scale;               // loss scale / batch
    int accumulate_latents;
}

float hgelu_derivative(float y)
{
    if (y > 3.0 || y <= -1.5)
        return 0.0;
    return y < 1.5 ? (2.0 / 3.0) * y + 0.5 : 1.0;
}

void add_latent_grad(NtcNetwork net, NtcTrainState st, int n, int x, int y, int f, float g)
{
    let grid = net.neural_mips[n];
    let element = grid.z + f * grid.x * grid.y + y * grid.x + x;
    InterlockedAdd(st.latent_grad[element], int64_t(round(g * LatentGradScale)));
}

void mark_texel(NtcNetwork net, NtcTrainState st, int n, int x, int y)
{
    let pixel = uint(st.pixel_offsets[n] + y * net.neural_mips[n].x + x);
    InterlockedOr(st.grad_mask[pixel >> 5], 1u << (pixel & 31));
}

void backward_grid(NtcNetwork net, NtcTrainState st, int n, float u, float v, float grad_inputs[NtcInputs], int base)
{
    let grid = net.neural_mips[n];
    let x = fma(u, float(grid.x), -0.5);
    let y = fma(v, float(grid.y), -0.5);
    int xb = int(floor(x));
    int yb = int(floor(y));
    let dx = x - xb;
    let dy = y - yb;
    float w[4] = { round_half((1 - dx) * (1 - dy)), round_half(dx * (1 - dy)),
                   round_half((1 - dx) * dy), round_half(dx * dy) };
    if (xb < 0) xb += grid.x;
    if (yb < 0) yb += grid.y;
    int xs[4] = { xb % grid.x, (xb + 1) % grid.x, xb % grid.x, (xb + 1) % grid.x };
    int ys[4] = { yb % grid.y, yb % grid.y, (yb + 1) % grid.y, (yb + 1) % grid.y };
    for (int c = 0; c < 4; ++c)
    {
        mark_texel(net, st, n, xs[c], ys[c]);
        for (int f = 0; f < net.num_features; ++f)
            add_latent_grad(net, st, n, xs[c], ys[c], f, grad_inputs[base + f] * 2.0 * w[c]);
    }
}

struct NtcTrainForward
{
    CoopVec<half, 48> x0;
    CoopVec<half, 64> y0;
    CoopVec<half, 64> x1;
    CoopVec<half, 48> y1;
    CoopVec<half, 48> x2;
    CoopVec<half, 32> y2;
    CoopVec<half, 32> x3;
    CoopVec<half, 16> y3;
}

CoopVec<half, N> layer_forward<let K : int, let N : int>(NtcTrainState st, int l, CoopVec<half, K> x)
{
    return coopVecMatMulAdd<half, N>(x, CoopVecComponentType.Float16, st.coop_weights, st.coop_weight_offsets[l],
                                     CoopVecComponentType.Float16, st.coop_biases, st.coop_bias_offsets[l],
                                     CoopVecComponentType.Float16, CoopVecMatrixLayout.TrainingOptimal, false, 0);
}

CoopVec<half, N> activate<let N : int>(NtcNetwork net, CoopVec<half, N> y, bool int8)
{
    CoopVec<half, N> x;
    for (int o = 0; o < N; ++o)
        x[o] = half(hgelu_clamp(float(y[o]), int8, net.one));
    return x;
}

// Weight and bias gradients of layer ``l`` added to the warp's slot; returns the input gradient.
CoopVec<half, K> layer_backward<let K : int, let N : int>(NtcTrainState st, int l, int t, CoopVec<half, K> x,
                                                         CoopVec<half, N> g)
{
    let slot = uint(t / 32) * uint(st.slot_bytes);
    coopVecOuterProductAccumulate(g, x, st.grad_slots, slot + st.coop_weight_offsets[l], 0,
                                  CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float16);
    coopVecReduceSumAccumulate(g, st.grad_slots, slot + st.slot_bias_start + st.coop_bias_offsets[l]);
    return coopVecMatMul<half, K>(g, CoopVecComponentType.Float16, st.coop_weights, st.coop_weight_offsets[l],
                                  CoopVecComponentType.Float16, CoopVecMatrixLayout.TrainingOptimal, true, 0);
}

CoopVec<half, K> through_activation<let K : int>(CoopVec<half, K> d, CoopVec<half, K> y)
{
    CoopVec<half, K> g;
    for (int k = 0; k < K; ++k)
        g[k] = half(float(d[k]) * hgelu_derivative(float(y[k])));
    return g;
}

// Forward for batch texel ``t``, keeping layer inputs and pre-activations for the backward pass.
NtcTrainForward forward_texel(NtcNetwork net, NtcTrainState st, int t)
{
    let texel = st.texels[t];
    let color_mip = net.color_mips[texel.x];
    let u = (float(texel.y) + 0.5) / float(color_mip.x);
    let v = (float(texel.z) + 0.5) / float(color_mip.y);

    float inputs[NtcInputs] = {};
    sample_grid(net, color_mip.z, u, v, 0, inputs);
    sample_grid(net, color_mip.z + 1, u, v, NtcFeaturesPerMip, inputs);
    let position = net.positions[texel.x];
    encode_position(float(texel.y) * position.x, float(texel.z) * position.x, position.y, inputs);

    NtcTrainForward f = {};
    for (int i = 0; i < NtcInputs; ++i)
        f.x0[i] = half(quantize_input(inputs[i], net.stage, net.one));
    let int8 = net.stage == NtcStageInt8;
    f.y0 = layer_forward<48, 64>(st, 0, f.x0);
    f.x1 = activate(net, f.y0, int8);
    f.y1 = layer_forward<64, 48>(st, 1, f.x1);
    f.x2 = activate(net, f.y1, int8);
    f.y2 = layer_forward<48, 32>(st, 2, f.x2);
    f.x3 = activate(net, f.y2, true);
    f.y3 = layer_forward<32, 16>(st, 3, f.x3);
    return f;
}

// Linear-space latent ``c`` from the network output ``o``, as the SDK's loss and decompression.
float linear_latent(NtcNetwork net, float o, int c)
{
    return round_half(fma(isfinite(o) ? o : 0.0, net.mapping[c].x, net.mapping[c].y));
}

// Backward for batch texel ``t`` from the gradient ``g_out`` at the network outputs
// (already scaled by loss scale / batch): records it, adds weight gradients to the
// warp's slot and latent gradients when ``accumulate_latents``.
void backward_texel(NtcNetwork net, NtcTrainState st, NtcLossParams lp, int t, NtcTrainForward f,
                    float g_out[NtcOutputs])
{
    CoopVec<half, 16> g3;
    for (int o = 0; o < NtcOutputs; ++o)
    {
        st.output_grads[t * NtcOutputs + o] = g_out[o];
        g3[o] = half(g_out[o]);
    }
    let g2 = through_activation(layer_backward<32, 16>(st, 3, t, f.x3, g3), f.y2);
    let g1 = through_activation(layer_backward<48, 32>(st, 2, t, f.x2, g2), f.y1);
    let g0 = through_activation(layer_backward<64, 48>(st, 1, t, f.x1, g1), f.y0);
    let d = layer_backward<48, 64>(st, 0, t, f.x0, g0);

    if (lp.accumulate_latents != 0)
    {
        let texel = st.texels[t];
        let color_mip = net.color_mips[texel.x];
        let u = (float(texel.y) + 0.5) / float(color_mip.x);
        let v = (float(texel.z) + 0.5) / float(color_mip.y);
        float grad_inputs[NtcInputs] = {};
        for (int i = 0; i < NtcInputs; ++i)
            grad_inputs[i] = float(d[i]);
        backward_grid(net, st, color_mip.z, u, v, grad_inputs, 0);
        backward_grid(net, st, color_mip.z + 1, u, v, grad_inputs, NtcFeaturesPerMip);
    }
}

// SDK L2 loss against the reference pyramid for batch texel ``t``. The gradient
// reaches the outputs without the channel mapping scale, as in the SDK.
void ntc_train_texel(NtcNetwork net, NtcTrainState st, NtcLossParams lp, int t)
{
    let f = forward_texel(net, st, t);
    let texel = st.texels[t];
    let ref = st.reference_offsets[texel.x] + (texel.z * net.color_mips[texel.x].x + texel.y) * NtcLatents;
    float g[NtcOutputs] = {};
    float loss = 0;
    for (int c = 0; c < NtcLatents; ++c)
    {
        let diff = linear_latent(net, float(f.y3[c]), c) - st.reference[ref + c];
        loss += diff * diff;
        g[c] = 2.0 * diff * lp.gradient_scale;
    }
    st.texel_loss[t] = loss;
    backward_texel(net, st, lp, t, f, g);
}

// Linear-space latents of batch texel ``t`` as training computes them; needs ``prepare_network``.
float[NtcLatents] ntc_train_latents(NtcNetwork net, NtcTrainState st, int t)
{
    let f = forward_texel(net, st, t);
    float latents[NtcLatents] = {};
    for (int c = 0; c < NtcLatents; ++c)
        latents[c] = linear_latent(net, float(f.y3[c]), c);
    return latents;
}

// Linear-space latents of batch texel ``t`` by the SDK decompression forward (``ntc_forward.slang``).
float[NtcLatents] ntc_linear_latents(NtcNetwork net, NtcTrainState st, int t)
{
    let texel = st.texels[t];
    float outputs[NtcOutputs];
    forward(net, texel.x, texel.yz, outputs);
    float latents[NtcLatents] = {};
    for (int c = 0; c < NtcLatents; ++c)
        latents[c] = linear_latent(net, outputs[c], c);
    return latents;
}

float slot_value(NtcTrainState st, int slot, int index)
{
    let offset = uint(slot) * uint(st.slot_bytes) + uint(index) * 2;
    let word = st.grad_slots.Load(offset & ~3u);
    return f16tof32((offset & 2) != 0 ? word >> 16 : word & 0xffff);
}

// Partial gradient of parameter ``id.x`` over batch chunk ``id.y``: its warps' slots summed in order.
void ntc_weight_chunk(NtcTrainState st, int2 id)
{
    let p = id.x;
    let c = id.y;
    let index = st.slot_index[p];
    let warps = st.chunk / 32;
    float s = 0;
    for (int w = c * warps; w < (c + 1) * warps; ++w)
        s += slot_value(st, w, index);
    st.partials[c * NtcParams + p] = s;
}

// ``accumulate``: add to the gradient of the step's earlier batches.
void ntc_weight_sum(NtcTrainState st, int accumulate, int p)
{
    float s = 0;
    for (int c = 0; c < st.batch / st.chunk; ++c)
        s += st.partials[c * NtcParams + p];
    st.weight_grad[p] = accumulate != 0 ? st.weight_grad[p] + s : s;
}

// Constants of ``PrepareAdamConstants`` and the update kind.
struct NtcAdam
{
    float inv_loss_scale;
    float beta1;
    float beta2;
    float epsilon;
    float step_size;
    float inv_sqrt_bias_correction2;
    uint seed;
    int half_offset;
}

// ``AdamOptimizerCore``. Returns false when the gradient is zero (no update, no random number).
bool adam(NtcAdam a, float gradient, inout float base, out float quantized, inout float m1, inout float m2,
          bool latent, inout HashRng rng)
{
    quantized = 0;
    if (gradient == 0.0)
        return false;
    m1 = a.beta1 * m1 + (1.0 - a.beta1) * gradient;
    m2 = a.beta2 * m2 + (1.0 - a.beta2) * (gradient * gradient);
    let denom = sqrt(m2) * a.inv_sqrt_bias_correction2 + a.epsilon;
    var w = base - (m1 / denom) * a.step_size;
    var q = round_half(w);
    if (latent)
    {
        w = clamp(w, 0.0, 1.0);
        let noise = rng.next_float() - 0.5;
        q = round_half(clamp(w + noise * LatentStep, 0.0, 1.0));
    }
    if (!isfinite(w))
        w = 0;
    if (!isfinite(q))
        q = 0;
    base = round_half(w);
    quantized = q;
    return true;
}

void ntc_network_adam(NtcTrainState st, NtcAdam a, int p)
{
    let i = a.half_offset + p;
    var base = st.weights_base[i];
    var m1 = st.weight_m1[p];
    var m2 = st.weight_m2[p];
    var rng = HashRng(a.seed, uint(p));
    float q;
    if (adam(a, st.weight_grad[p] * a.inv_loss_scale, base, q, m1, m2, false, rng))
    {
        st.weights_base[i] = base;
        st.weights_q[i] = q;
        st.weight_m1[p] = m1;
        st.weight_m2[p] = m2;
    }
}

// Latent pixel ``id.x`` (SDK order over all neural mips), feature pair ``id.y``.
void ntc_latent_adam(NtcNetwork net, NtcTrainState st, NtcAdam a, int2 id)
{
    let pixel = id.x;
    let group = id.y;
    if ((st.grad_mask[uint(pixel) >> 5] & (1u << (uint(pixel) & 31))) == 0)
        return;
    int n = 0;
    while (n + 1 < st.num_neural_mips && pixel >= st.pixel_offsets[n + 1])
        ++n;
    let grid = net.neural_mips[n];
    let plane = grid.x * grid.y;
    let local = pixel - st.pixel_offsets[n];
    var rng = HashRng(a.seed, uint(pixel * 2 + group * st.total_pixels * 2));
    for (int j = 0; j < 2; ++j)
    {
        let e = grid.z + (2 * group + j) * plane + local;
        let gradient = float(st.latent_grad[e]) * (1.0 / LatentGradScale) * a.inv_loss_scale;
        st.latent_grad[e] = 0;
        var base = st.latents_base[e];
        var m1 = st.latent_m1[e];
        var m2 = st.latent_m2[e];
        float q;
        if (adam(a, gradient, base, q, m1, m2, true, rng))
        {
            st.latents_base[e] = base;
            st.latents_q[e] = q;
            st.latent_m1[e] = m1;
            st.latent_m2[e] = m2;
        }
    }
}

// ``FreezeQuantization``: quantized latents rounded to the 4-bit grid, base unchanged.
void ntc_freeze(NtcTrainState st, int e)
{
    let w = clamp(st.latents_base[e], 0.0, 1.0);
    st.latents_q[e] = round_half(round_away(w / LatentStep) * LatentStep);
}

float round_away(float v)
{
    return sign(v) * floor(abs(v) + 0.5);
}

// a / b rounded to nearest, as CUDA's IEEE division and ``__frcp_rn``. GPU division may be off
// by more than half an ulp; the neighbor with the smallest exact residual a - b * q is chosen.
float divide_rn(float a, float b)
{
    let q0 = a / b;
    var best = q0;
    var best_residual = abs(fma(-b, q0, a));
    let up = asfloat(asuint(q0) + 1u);
    let down = asfloat(asuint(q0) - 1u);
    for (int i = 0; i < 2; ++i)
    {
        let q = i == 0 ? up : down;
        let r = abs(fma(-b, q, a));
        if (r < best_residual)
        {
            best = q;
            best_residual = r;
        }
    }
    return best;
}

// ``QuantizeNetwork`` without output: column ``col`` (all layers' outputs in order) of the
// weights from ``half_offset``, rounded in place to FP8 (``fp8`` and not layer 3) or to the
// layer's Int8 column grid.
void ntc_fake_quantize(NtcTrainState st, int half_offset, int fp8, int col)
{
    int layer = 0;
    int first = 0;
    while (col >= first + NtcLayerOutputs[layer])
    {
        first += NtcLayerOutputs[layer];
        ++layer;
    }
    let o = col - first;
    let n_in = NtcLayerInputs[layer];
    let row = half_offset + NtcLayerWeightOffsets[layer] + o * n_in;
    if (fp8 != 0 && layer < 3)
    {
        for (int k = 0; k < n_in; ++k)
            st.weights_q[row + k] = round_fp8(st.weights_q[row + k]);
        return;
    }
    float lo = 3.402823466e38;
    float hi = 1.175494351e-38;
    for (int k = 0; k < n_in; ++k)
    {
        lo = min(lo, st.weights_q[row + k]);
        hi = max(hi, st.weights_q[row + k]);
    }
    let limit = max(abs(hi), abs(lo));
    let ilimit = divide_rn(1.0, limit);
    let iscale = divide_rn(1.0, 127.5);
    for (int k = 0; k < n_in; k += 2)
    {
        let qx = clamp(round_away(st.weights_q[row + k] * 127.5 * ilimit), -127.0, 127.0);
        let qy = clamp(round_away(st.weights_q[row + k + 1] * divide_rn(127.5, limit)), -127.0, 127.0);
        st.weights_q[row + k] = round_half(qx * limit * iscale);
        st.weights_q[row + k + 1] = round_half(qy * limit * iscale);
    }
}

void ntc_clear_mask(NtcTrainState st, int i)
{
    st.grad_mask[i] = 0;
}

// FP16 biases of the weight set from ``half_offset`` into ``biases``, two per thread: bias pair ``i``
// in the parameter order of ``weights_q``, each layer's vector from ``coop_bias_offsets``.
void ntc_pack_biases(NtcTrainState st, int half_offset, RWByteAddressBuffer biases, int i)
{
    let p = 2 * i;
    int layer = 3;
    while (p < NtcLayerBiasOffsets[layer] - NtcBiasStart)
        --layer;
    let o = p - (NtcLayerBiasOffsets[layer] - NtcBiasStart);
    let a = f32tof16(st.weights_q[half_offset + NtcBiasStart + p]);
    let b = f32tof16(st.weights_q[half_offset + NtcBiasStart + p + 1]);
    biases.Store(uint(st.coop_bias_offsets[layer] + o * 2), a | (b << 16));
}
