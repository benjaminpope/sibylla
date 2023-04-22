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
        B, H, W, C = y.shape
        x=y.reshape(B, H, W, 2, 2, C//4).transpose((0, 1, 3, 2, 4, 5)).reshape(B, H*2, W*2, C//4)
        logdet = 0
        return x, logdet


class IgnorantMaskedCoupling(base.Bijector):
    """
    An implementation of distrax MaskedCoupling but also includes a seperate mask `exclude`
    which are ignored by the coupler

    Let `f` be a conditional bijector (the inner bijector), `g` be a function (the
    conditioner), and `m`, `n` be boolean masks interpreted numerically, such that
    True is 1 and False is 0. `m` represents the coupling mask, while `n` represents the ignorance mask.
    The ignorant masked coupling bijector is defined as follows:

    - Forward: `y = (1-m-n) * f(x; g(m*x)) + (m+n)*x`

    - Forward Jacobian log determinant:
        `log|det J(x)| = sum((1-m-n) * log|df/dx(x; g(m*x))|)`

    - Inverse: `x = (1-m-n) * f^{-1}(y; g(m*y)) + (m+n)*y`

    - Inverse Jacobian log determinant:
        `log|det J(y)| = sum((1-m-n) * log|df^{-1}/dy(y; g(m*y))|)`

    This is taken by relations from the MaskedCoupling but instead having (z=x or y):
        z_conditioning = m*z -> m*z # the subset used for conditioning
        z_edited = (1-m)*z -> (1-m-n)*z # the subset that gets changed
    and introducing
        z_ignored = n*z
    """
    def __init__(self, 
                 coupling_mask : Array,
                 ignorance_mask : Array):
        
        if np.logical_or(coupling_mask, ignorance_mask).any():
            raise ValueError('The masks must have no overlap')

        self.coupling_mask = coupling_mask
        self.ignorance_mask = ignorance_mask


    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = x
        logdet = 0
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x=y
        logdet = 0
        return x, logdet