// SPDX-License-Identifier: Apache-2.0

// Forward pass of the NTC SDK's CUDA training and decompression kernels
// (``RegressionKernels.h``, ``FeatureGridDevice.h``, ``CudaMLP.h``, ``tin``),
// in float32 with FP16 and FP8 rounding at the SDK's rounding points.
// Tensor-core layers start from the bias and round to FP16 once per 8 inputs,
// from the exact sum. Half operations fused by nvcc round once, from the exact
// result. Values of FP16 type are carried as float32 holding FP16 values.
import slangpy;

static const int NtcMaxColorMips = 16;
static const int NtcMaxNeuralMips = 8;
static const int NtcFeaturesPerMip = 16;
static const int NtcInputs = 48;
static const int NtcMaxWidth = 64;
static const int NtcOutputs = 16;
static const int NtcLatents = 8;
static const int NtcEncodingOffset = 32;
static const int NtcEncodingWaves = 3;
static const int NtcChunk = 8;

static const int NtcStageFP8 = 0;
static const int NtcStageInt8 = 1;

static const int NtcLayerInputs[4] = { 48, 64, 48, 32 };
static const int NtcLayerOutputs[4] = { 64, 48, 32, 16 };

// One latent grid set, decode network and channel mapping.
// Latents: per neural mip n, ``features`` planes of ``width * height`` values
// from element ``neural_mips[n].z``. Weights: per layer a row-major
// (outputs, inputs) matrix from ``weight_offsets[l]`` and a bias vector from
// ``bias_offsets[l]``.
struct NtcNetwork
{
    StructuredBuffer<float> latents;
    StructuredBuffer<float> weights;
    int4 color_mips[NtcMaxColorMips];   // width, height, neural mip
    float2 positions[NtcMaxColorMips];  // position scale, position lod
    int4 neural_mips[NtcMaxNeuralMips]; // width, height, first element
    int4 weight_offsets;
    int4 bias_offsets;
    float2 mapping[NtcOutputs];         // linear = optimal * x + y
    int num_features;
    int stage;
    float one;                          // 1.0, opaque to the compiler (see two_sum)
}

float round_even(float v);

// Nearest FP16 value, ties to even, with FP16 subnormals and overflow to infinity.
// In software: the f32tof16 and f16tof32 pair is folded away by the compiler.
float round_half(float x)
{
    let a = abs(x);
    if (isnan(a))
        return x;
    let e = max(int((asuint(a) >> 23) & 0xff) - 127, -14);
    let quantum = ldexp(1.0, e - 10);
    var r = round_even(a / quantum) * quantum;
    if (r > 65504.0)
        r = asfloat(0x7f800000u);
    return (asuint(x) >> 31) != 0 ? -r : r;
}

float round_even(float v)
{
    let f = floor(v);
    let d = v - f;
    if (d > 0.5)
        return f + 1;
    if (d < 0.5)
        return f;
    return (fmod(f, 2.0) == 0.0) ? f : f + 1;
}

// FP16 value to the nearest FP8 E4M3 value, ties to even, saturating at 448.
float round_fp8(float x)
{
    let a = min(abs(x), 448.0);
    if (a == 0.0)
        return 0.0;
    let e = max(int((asuint(a) >> 23) & 0xff) - 127, -6);
    let quantum = ldexp(1.0, e - 3);
    return sign(x) * min(round_even(a / quantum) * quantum, 448.0);
}

// a + b exactly, as a float32 sum and its rounding error. ``one`` is 1.0 from
// a uniform: without it the compiler folds the error term to zero.
void two_sum(float a, float b, float one, out float s, out float e)
{
    s = (a + b) * one;
    let bb = (s - a) * one;
    e = (a - (s - bb)) + (b - bb);
}

// FP16 rounding of the exact value s + e, with |e| below half a float32 ulp of s.
// Only an s exactly halfway between two FP16 values needs e.
float round_half_sum(float s, float e)
{
    let r = round_half(s);
    let a = abs(s);
    if (e == 0.0 || a == 0.0 || isinf(a))
        return r;
    let quantum = ldexp(1.0, max(int((asuint(a) >> 23) & 0xff) - 127, -14) - 10);
    let t = a / quantum;
    if (t - floor(t) != 0.5)
        return r;
    var m = ((e > 0.0) == (s > 0.0) ? floor(t) + 1 : floor(t)) * quantum;
    if (m > 65504.0)
        m = asfloat(0x7f800000u);
    return s < 0.0 ? -m : m;
}

// Fused FP16 multiply-add: a * b is exact in float32 for FP16 a and b.
float hfma(float a, float b, float c, float one)
{
    float s, e;
    two_sum(a * b, c, one, s, e);
    return round_half_sum(s, e);
}

void sample_grid(NtcNetwork net, int n, float u, float v, int base, inout float inputs[NtcInputs])
{
    let grid = net.neural_mips[n];
    let x = fma(u, float(grid.x), -0.5);
    let y = fma(v, float(grid.y), -0.5);
    int xb = int(floor(x));
    int yb = int(floor(y));
    let dx = x - xb;
    let dy = y - yb;
    let w0 = round_half((1 - dx) * (1 - dy));
    let w1 = round_half(dx * (1 - dy));
    let w2 = round_half((1 - dx) * dy);
    let w3 = round_half(dx * dy);
    if (xb < 0) xb += grid.x;
    if (yb < 0) yb += grid.y;
    let x0 = xb % grid.x;
    let y0 = yb % grid.y;
    let x1 = (xb + 1) % grid.x;
    let y1 = (yb + 1) % grid.y;
    let plane = grid.x * grid.y;
    for (int f = 0; f < net.num_features; ++f)
    {
        let p = grid.z + f * plane;
        var d = round_half(net.latents[p + y0 * grid.x + x1] * w1);
        d = hfma(net.latents[p + y0 * grid.x + x0], w0, d, net.one);
        d = hfma(net.latents[p + y1 * grid.x + x0], w2, d, net.one);
        d = hfma(net.latents[p + y1 * grid.x + x1], w3, d, net.one);
        inputs[base + f] = hfma(d, 2, -1, net.one);
    }
}

void encode_position(float xf, float yf, float lod, inout float inputs[NtcInputs])
{
    int idx = NtcEncodingOffset;
    for (int wave = 0; wave < NtcEncodingWaves; ++wave)
    {
        inputs[idx++] = round_half(frac(xf) * 2 - 1);
        inputs[idx++] = round_half(frac(yf) * 2 - 1);
        inputs[idx++] = round_half(frac(xf + 0.25) * 2 - 1);
        inputs[idx++] = round_half(frac(yf + 0.25) * 2 - 1);
        xf *= 2;
        yf *= 2;
    }
    inputs[idx++] = round_half(lod);
    inputs[idx++] = round_half(lod);
}

float quantize_input(float x, int stage, float one)
{
    if (stage == NtcStageFP8)
        return round_fp8(x);
    let y = clamp(floor(hfma(x, 127.5, -0.5, one)), -127.0, 127.0);
    return round_half(y * round_half(1.0 / 127.5));
}

float hgelu_clamp(float x, bool int8, float one)
{
    let t = saturate(hfma(round_half(1.0 / 3.0), x, 0.5, one));
    let y = round_half(min(x, 3.0) * t);
    if (!int8)
        return round_fp8(y);
    return round_half(round_even(round_half(y * 80.0)) * round_half(0.0125));
}

void dense_layer(NtcNetwork net, int l, float x[NtcMaxWidth], out float y[NtcMaxWidth])
{
    let n_in = NtcLayerInputs[l];
    let n_out = NtcLayerOutputs[l];
    let w = net.weight_offsets[l];
    let b = net.bias_offsets[l];
    y = {};
    for (int o = 0; o < n_out; ++o)
    {
        var acc = net.weights[b + o];
        for (int k0 = 0; k0 < n_in; k0 += NtcChunk)
        {
            float s = acc;
            float e = 0;
            for (int k = k0; k < k0 + NtcChunk; ++k)
            {
                float err;
                two_sum(s, net.weights[w + o * n_in + k] * x[k], net.one, s, err);
                e += err;
            }
            two_sum(s, e, net.one, s, e);
            acc = round_half_sum(s, e);
        }
        y[o] = acc;
    }
}

// Network outputs in the [0, 1] space at integer texel ``texel`` of color mip ``mip``.
void forward(NtcNetwork net, int mip, int2 texel, out float outputs[NtcOutputs])
{
    let color_mip = net.color_mips[mip];
    let u = (float(texel.x) + 0.5) / float(color_mip.x);
    let v = (float(texel.y) + 0.5) / float(color_mip.y);

    float inputs[NtcInputs] = {};
    sample_grid(net, color_mip.z, u, v, 0, inputs);
    sample_grid(net, color_mip.z + 1, u, v, NtcFeaturesPerMip, inputs);
    let position = net.positions[mip];
    encode_position(float(texel.x) * position.x, float(texel.y) * position.x, position.y, inputs);

    float x[NtcMaxWidth] = {};
    for (int i = 0; i < NtcInputs; ++i)
        x[i] = quantize_input(inputs[i], net.stage, net.one);

    float y[NtcMaxWidth];
    for (int l = 0; l < 4; ++l)
    {
        dense_layer(net, l, x, y);
        if (l < 3)
        {
            let int8 = net.stage == NtcStageInt8 || l == 2;
            for (int o = 0; o < NtcLayerOutputs[l]; ++o)
                x[o] = hgelu_clamp(y[o], int8, net.one);
        }
    }
    for (int o = 0; o < NtcOutputs; ++o)
        outputs[o] = y[o];
}

// Linear-space latents at ``texel`` of color mip ``mip``, as the SDK decompression writes them.
void ntc_forward_mip(NtcNetwork net, int mip, int2 texel, out float4 a, out float4 b)
{
    float outputs[NtcOutputs];
    forward(net, mip, texel, outputs);
    float linear[NtcLatents] = {};
    for (int c = 0; c < NtcLatents; ++c)
    {
        let o = isfinite(outputs[c]) ? outputs[c] : 0.0;
        linear[c] = round_half(fma(o, net.mapping[c].x, net.mapping[c].y));
    }
    a = float4(linear[0], linear[1], linear[2], linear[3]);
    b = float4(linear[4], linear[5], linear[6], linear[7]);
}

float ntc_round_half(float x)
{
    return round_half(x);
}

float ntc_round_fp8(float x)
{
    return round_fp8(x);
}

float ntc_hfma(float a, float b, float c, float one)
{
    return hfma(a, b, c, one);
}
