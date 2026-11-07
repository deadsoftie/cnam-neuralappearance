// SPDX-License-Identifier: Apache-2.0

// One training step of the NTC SDK compression (``TextureSet.cpp``,
// ``RegressionKernels.h``, ``Optimizer.cu``, ``Quantizer.cu``) as separate
// kernels. Forward as ``ntc_forward.slang``; backward in float32 with
// straight-through quantization. Weight gradients are summed in texel order;
// latent gradients are summed as 64-bit fixed point (``LatentGradScale``), so
// every reduction is independent of thread order.
import slangpy;
__exported import ntc_training.ntc_forward;

static const int NtcParams = 8352;
static const int NtcBiasStart = 8192;
static const int NtcActs = 192;
static const int NtcGrads = 160;
static const int NtcLayerWeightOffsets[4] = { 0, 3072, 6144, 7680 };
static const int NtcLayerBiasOffsets[4] = { 8192, 8256, 8304, 8336 };
static const int NtcActOffsets[4] = { 0, 48, 112, 160 };
static const int NtcGradOffsets[4] = { 0, 64, 112, 144 };
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
    RWStructuredBuffer<float> acts;
    RWStructuredBuffer<float> grads;
    RWStructuredBuffer<float> texel_loss;
    RWStructuredBuffer<float> partials;
    StructuredBuffer<float> reference;  // per color mip, 8 channels per texel
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

// Forward, L2 loss against the reference and backward for batch texel ``t``.
void ntc_train_texel(NtcNetwork net, NtcTrainState st, NtcLossParams lp, int t)
{
    let texel = st.texels[t];
    let mip = texel.x;
    let color_mip = net.color_mips[mip];
    let u = (float(texel.y) + 0.5) / float(color_mip.x);
    let v = (float(texel.z) + 0.5) / float(color_mip.y);

    float inputs[NtcInputs] = {};
    sample_grid(net, color_mip.z, u, v, 0, inputs);
    sample_grid(net, color_mip.z + 1, u, v, NtcFeaturesPerMip, inputs);
    let position = net.positions[mip];
    encode_position(float(texel.y) * position.x, float(texel.z) * position.x, position.y, inputs);

    float x[NtcMaxWidth] = {};
    for (int i = 0; i < NtcInputs; ++i)
        x[i] = quantize_input(inputs[i], net.stage, net.one);

    float pre[4][NtcMaxWidth];
    float y[NtcMaxWidth];
    let acts = t * NtcActs;
    for (int l = 0; l < 4; ++l)
    {
        for (int k = 0; k < NtcLayerInputs[l]; ++k)
            st.acts[acts + NtcActOffsets[l] + k] = x[k];
        dense_layer(net, l, x, y);
        pre[l] = y;
        if (l < 3)
        {
            let int8 = net.stage == NtcStageInt8 || l == 2;
            for (int o = 0; o < NtcLayerOutputs[l]; ++o)
                x[o] = hgelu_clamp(y[o], int8, net.one);
        }
    }

    float g[NtcMaxWidth] = {};
    float loss = 0;
    let ref = st.reference_offsets[mip] + (texel.z * color_mip.x + texel.y) * NtcLatents;
    for (int c = 0; c < NtcLatents; ++c)
    {
        let o = isfinite(pre[3][c]) ? pre[3][c] : 0.0;
        let diff = round_half(fma(o, net.mapping[c].x, net.mapping[c].y)) - st.reference[ref + c];
        loss += diff * diff;
        g[c] = 2.0 * diff * lp.gradient_scale;
    }
    st.texel_loss[t] = loss;

    let grads = t * NtcGrads;
    for (int l = 3; l >= 0; --l)
    {
        let n_in = NtcLayerInputs[l];
        let n_out = NtcLayerOutputs[l];
        for (int o = 0; o < n_out; ++o)
            st.grads[grads + NtcGradOffsets[l] + o] = g[o];
        float back[NtcMaxWidth] = {};
        let w = net.weight_offsets[l];
        for (int k = 0; k < n_in; ++k)
        {
            float s = 0;
            for (int o = 0; o < n_out; ++o)
                s += net.weights[w + o * n_in + k] * g[o];
            back[k] = l > 0 ? s * hgelu_derivative(pre[l - 1][k]) : s;
        }
        g = back;
    }

    if (lp.accumulate_latents != 0)
    {
        float grad_inputs[NtcInputs] = {};
        for (int i = 0; i < NtcInputs; ++i)
            grad_inputs[i] = g[i];
        backward_grid(net, st, color_mip.z, u, v, grad_inputs, 0);
        backward_grid(net, st, color_mip.z + 1, u, v, grad_inputs, NtcFeaturesPerMip);
    }
}

// Partial gradient of parameter ``id.x`` over batch chunk ``id.y``, summed in texel order.
void ntc_weight_chunk(NtcTrainState st, int2 id)
{
    let p = id.x;
    let c = id.y;
    let bias = p >= NtcBiasStart;
    int layer = 3;
    while (p < (bias ? NtcLayerBiasOffsets[layer] : NtcLayerWeightOffsets[layer]))
        --layer;
    let n_in = NtcLayerInputs[layer];
    let o = bias ? p - NtcLayerBiasOffsets[layer] : (p - NtcLayerWeightOffsets[layer]) / n_in;
    let k = bias ? 0 : (p - NtcLayerWeightOffsets[layer]) % n_in;
    float s = 0;
    for (int t = c * st.chunk; t < (c + 1) * st.chunk; ++t)
    {
        let g = st.grads[t * NtcGrads + NtcGradOffsets[layer] + o];
        s += bias ? g : g * st.acts[t * NtcActs + NtcActOffsets[layer] + k];
    }
    st.partials[c * NtcParams + p] = s;
}

void ntc_weight_sum(NtcTrainState st, int p)
{
    float s = 0;
    for (int c = 0; c < st.batch / st.chunk; ++c)
        s += st.partials[c * NtcParams + p];
    st.weight_grad[p] = s;
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
    m2 = a.beta2 * m2 + (1.0 - a.beta2) * gradient * gradient;
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
