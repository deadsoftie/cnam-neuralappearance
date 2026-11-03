# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import slangpy as spy

from ..basetypes import Real
from .optimizer import Optimizer


class RegularizedAdamOptimizer(Optimizer):
    """Adam with L2 regularization and optional finite-gradient checks.

    Moments live in Slang. Python tracks the iteration and supplies
    bias-correction factors with each dispatch. Every non-finite gradient
    element is counted in ``invalid_gradient_counter``, whether or not
    ``ignore_invalid_gradients`` zeroes it.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_regularization: float = 0.0,
        ignore_invalid_gradients: bool = False,
        *,
        invalid_gradient_counter: spy.Buffer,
        **kwargs,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l2_regularization = l2_regularization
        self.ignore_invalid_gradients = ignore_invalid_gradients
        self.invalid_gradient_counter = invalid_gradient_counter
        self.iteration = 1

    @staticmethod
    def create_invalid_gradient_counter(device: spy.Device) -> spy.Buffer:
        return device.create_buffer(
            size=8,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            data=np.zeros(2, dtype=np.uint32),
        )

    def read_invalid_gradients(self) -> tuple[int, bool]:
        """Return the non-finite gradient count and whether any occurred."""
        count, seen = self.invalid_gradient_counter.to_numpy().view(np.uint32)[:2]
        return int(count), bool(seen)

    def update_state(self):
        self.iteration += 1

    def get_type_name(self, dtype: Real) -> str:
        return f'RegularizedAdamOptimizer<{dtype}>'

    def get_this(self):
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'l2_regularization': self.l2_regularization,
            'ignore_invalid_gradients': self.ignore_invalid_gradients,
            'invalid_gradient_counter': self.invalid_gradient_counter.descriptor_handle_rw,
            'mean_correction_factor': 1.0 / (1.0 - self.beta1**self.iteration),
            'variance_correction_factor': 1.0 / (1.0 - self.beta2**self.iteration),
        }
