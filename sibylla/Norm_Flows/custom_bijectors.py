"""
Custom distrax bijectors for additional functionality
"""

import distrax._src.bijectors.bijector as base
import jax.numpy as np
from typing import Any, Callable, Optional, Tuple
from distrax._src.utils import conversion
from distrax._src.utils import math

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

    def __init__(self,
               event_ndims_in: int,
               event_ndims_out: int):
        super().__init__(event_ndims_in, event_ndims_out, is_constant_jacobian=False,is_constant_log_det=True)


    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        H, W, C = x.shape
        y = x.reshape(H//2, 2, W//2, 2, C).transpose((0, 2, 1, 3, 4)).reshape(H//2, W//2, 4*C)
        logdet = 0
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        H, W, C = y.shape
        x=y.reshape(H, W, 2, 2, C//4).transpose((0, 2, 1, 3, 4)).reshape(H*2, W*2, C//4)
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
                 ignorance_mask : Array,
                 conditioner: Callable[[Array], BijectorParams],
                 bijector: Callable[[BijectorParams], base.BijectorLike],
                 event_ndims: Optional[int] = None,
                 inner_event_ndims: int = 0):
        
        if np.logical_or(coupling_mask, ignorance_mask).any():
            raise ValueError('The masks must have no overlap')
        
        if coupling_mask.shape != ignorance_mask.shape:
            raise ValueError(f'Coupling mask and ignorance mask must have same shape, got {coupling_mask.shape}, {ignorance_mask.shape}')


        if ignorance_mask.dtype != bool:
            raise ValueError(f'`ignorance_mask` must have values of type `bool`; got values of'
                        f' type `{ignorance_mask.dtype}`.')
        if coupling_mask.dtype != bool:
            raise ValueError(f'`coupling_mask` must have values of type `bool`; got values of'
                        f' type `{coupling_mask.dtype}`.')
        if event_ndims is not None and event_ndims < inner_event_ndims:
            raise ValueError(f'`event_ndims={event_ndims}` should be at least as'
                        f' large as `inner_event_ndims={inner_event_ndims}`.')
        
        self._coupling_mask = coupling_mask
        self._ignorance_mask = ignorance_mask
        self._unchanged_mask = np.logical_or(coupling_mask, ignorance_mask)
        self._coupling_event_mask = np.reshape(coupling_mask, coupling_mask.shape + (1,) * inner_event_ndims)
        self._ignorance_event_mask = np.reshape(ignorance_mask, ignorance_mask.shape + (1,) * inner_event_ndims)
        self._unchanged_event_mask = np.logical_or(self._coupling_event_mask, self._ignorance_event_mask)
        self._conditioner = conditioner
        self._bijector = bijector
        self._inner_event_ndims = inner_event_ndims
        if event_ndims is None:
            self._event_ndims = ignorance_mask.ndim + inner_event_ndims
        else:
            self._event_ndims = event_ndims
        super().__init__(event_ndims_in=self._event_ndims)

    @property
    def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]:
        """The callable that returns the inner bijector of `MaskedCoupling`."""
        return self._bijector

    @property
    def conditioner(self) -> Callable[[Array], BijectorParams]:
        """The conditioner function."""
        return self._conditioner

    @property
    def coupling_mask(self) -> Array:
        """The mask characterizing the `MaskedCoupling`, with boolean `dtype`."""
        return self._coupling_mask
    
    @property
    def ignorance_mask(self) -> Array:
        """The mask characterizing the `MaskedCoupling`, with boolean `dtype`."""
        return self._ignorance_mask

    def _inner_bijector(self, params: BijectorParams) -> base.Bijector:
        bijector = conversion.as_bijector(self._bijector(params))
        if (bijector.event_ndims_in != self._inner_event_ndims
            or bijector.event_ndims_out != self._inner_event_ndims):
            raise ValueError(
                'The inner bijector event ndims in and out must match the'
                f' `inner_event_ndims={self._inner_event_ndims}`. Instead, got'
                f' `event_ndims_in={bijector.event_ndims_in}` and'
                f' `event_ndims_out={bijector.event_ndims_out}`.')
        return bijector


    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

        self._check_forward_input_shape(x)
        masked_x = np.where(self._coupling_event_mask, x, 0.)
        params = self._conditioner(masked_x)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = np.where(self._unchanged_event_mask, x, y0)
        logdet = math.sum_last(
            np.where(self._unchanged_mask, 0., log_d),
            self._event_ndims - self._inner_event_ndims)
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        masked_y = np.where(self._coupling_event_mask, y, 0.)
        params = self._conditioner(masked_y)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = np.where(self._unchanged_event_mask, y, x0)
        logdet = math.sum_last(np.where(self._unchanged_mask, 0., log_d),
                            self._event_ndims - self._inner_event_ndims)
        return x, logdet