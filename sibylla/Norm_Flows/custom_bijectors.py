"""
Custom distrax bijectors for additional functionality
"""

import distrax._src.bijectors.bijector as base
import jax.numpy as np
from typing import Any, Callable, Optional, Tuple


Array = base.Array
BijectorParams = Any

class Squeeze(base.Bijector):
    """
    A squeezing bijector that converts spatial information into channel information

    Currently assumes the input has three dimensions, and modifies the sizes of the dimensions such that:
        Forward direction: H x W x C => H/2 x W/2 x 4C
        Inverse direction: H/2 x W/2 x 4C => H x W x C

    In either case, the log determinant of the jacobian is 0
    """

    def __init__(self):
        pass


    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        B, H, W, C = x.shape
        y = x.reshape(B, H//2, 2, W//2, 2, C).transpose((0, 1, 3, 2, 4, 5)).reshape(B, H//2, W//2, 4*C)
        logdet = 0
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x=y
        logdet = 0
        return x, logdet

