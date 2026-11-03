// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;
import slangpy;

__include ioptimizer;

public struct RegularizedAdamOptimizer<T : IReal> : IOptimizer<T>
{
    // One pair of moving moments is retained for every scalar parameter.
    public struct AdamState<S : IReal> : IOptimizerState<S>
    {
        S mean;
        S variance;

        public __init(S param)
        {
            mean = S(0.0f);
            variance = S(0.0f);
        }
    }
    public typealias State = AdamState<T>;

    // Handles for one parameter tensor and its gradient and state tensors.
    public struct AdamBatch<S : IReal>
    {
        RWStructuredBuffer<S>.Handle params;
        RWStructuredBuffer<S>.Handle grads;
        RWStructuredBuffer<AdamState<S>>.Handle states;
    }
    public typealias Batch = AdamBatch<T>;

    // The host computes bias correction because it depends on the global
    // optimizer iteration rather than individual parameters.
    public T beta1;
    public T beta2;
    public T epsilon;
    public T learning_rate;
    public T mean_correction_factor;
    public T variance_correction_factor;
    public T l2_regularization;
    public bool ignore_invalid_gradients;
    // Word 0 counts non-finite gradients, word 1 is set once any occurs, so
    // a wrapped count cannot hide them.
    public RWByteAddressBuffer.Handle invalid_gradient_counter;

    public void step(inout State state, inout T param, inout T grad)
    {
        // Apply L2 regularization directly to the gradient.
        grad += l2_regularization * param;

        if (!isfinite(grad))
        {
            invalid_gradient_counter.InterlockedAdd(0, 1);
            invalid_gradient_counter.InterlockedOr(4, 1);
            if (ignore_invalid_gradients)
                grad = T(0.0f);
        }

        // Update the moving moments with ``lerp()``.
        state.mean = lerp(grad, state.mean, beta1);
        state.variance = lerp(grad * grad, state.variance, beta2);

        // Correct moment bias using factors computed on the host.
        T mean_corrected = state.mean * mean_correction_factor;
        T variance_corrected = state.variance * variance_correction_factor;

        param -= learning_rate * mean_corrected / (sqrt(variance_corrected) + epsilon);
        grad = T(0.0f);
    }

    public void batch_step(ITensor<Batch, 1> batches, int2 batch_index)
    {
        AdamBatch<T> batch = batches[batch_index.x];

        int i = batch_index.y;

        T param = batch.params[i];
        T grad = batch.grads[i];
        AdamState<T> state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }
}
