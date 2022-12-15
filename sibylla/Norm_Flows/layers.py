"""
layers for normalising flows
modified from
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial11/NF_image_modeling.html
"""

from typing import Sequence
import numpy as onp

# JAX
import jax
import jax.numpy as np
from jax import random

from flax import linen as nn

# ------------------------------- Dequantization -------------------------------


class Dequantization(nn.Module):
    alpha: float = 1e-5  # Small constant that is used to scale the original input for numerical stability.
    quants: int = 256  # Number of possible discrete values (usually 256 for 8-bit image)

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, ldj, rng = self.dequant(z, ldj, rng)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += onp.log(self.quants) * onp.prod(z.shape[1:])
            z = np.floor(z)
            z = jax.lax.clamp(min=0.0, x=z, max=self.quants - 1.0).astype(np.int32)
        return z, ldj, rng

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z - 2 * jax.nn.softplus(-z)).sum(axis=[1, 2, 3])
            z = nn.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= onp.log(1 - self.alpha) * onp.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += onp.log(1 - self.alpha) * onp.prod(z.shape[1:])
            ldj += (-np.log(z) - np.log(1 - z)).sum(axis=[1, 2, 3])
            z = np.log(z) - np.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj, rng):
        # Transform discrete values to continuous volumes
        z = z.astype(np.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        ldj -= onp.log(self.quants) * onp.prod(z.shape[1:])
        return z, ldj, rng


class VariationalDequantization(Dequantization):
    var_flows: Sequence[nn.Module] = None  # A list of flow transformations to use for modeling q(u|x)

    def dequant(self, z, ldj, rng):
        z = z.astype(np.float32)
        img = (z / 255.0) * 2 - 1  # We condition the flows on x, i.e. the original image

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        rng, uniform_rng = random.split(rng)
        deq_noise = random.uniform(uniform_rng, z.shape)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        if self.var_flows is not None:
            for flow in self.var_flows:
                deq_noise, ldj, rng = flow(deq_noise, ldj, rng, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / 256.0
        ldj -= onp.log(256.0) * onp.prod(z.shape[1:])
        return z, ldj, rng


# ------------------------------- Standard layers -------------------------------


class CouplingLayer(nn.Module):
    network: nn.Module  # NN to use in the flow for predicting mu and sigma
    mask: onp.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in: int  # Number of input channels

    def setup(self):
        self.scaling_factor = self.param('scaling_factor', nn.initializers.zeros, (self.c_in,))

    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(np.concatenate([z_in, orig_img], axis=-1))
        s, t = nn_out.split(2, axis=-1)

        # Stabilize scaling output
        s_fac = np.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * np.exp(s)
            ldj += s.sum(axis=[1, 2, 3])
        else:
            z = (z * np.exp(-s)) - t
            ldj -= s.sum(axis=[1, 2, 3])

        return z, ldj, rng


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return np.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """This module applies a two-layer convolutional ResNet block with input gate"""

    c_in: int  # Number of input channels
    c_hidden: int  # Number of hidden dimensions

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential(
            [
                ConcatELU(),
                nn.Conv(self.c_hidden, kernel_size=(3, 3)),
                ConcatELU(),
                nn.Conv(2 * self.c_in, kernel_size=(1, 1)),
            ]
        )(x)
        val, gate = out.split(2, axis=-1)
        return x + val * nn.sigmoid(gate)


class GatedConvNet(nn.Module):
    c_hidden: int  # Number of hidden dimensions to use within the network
    c_out: int  # Number of output channels
    num_layers: int = 3  # Number of gated ResNet blocks to apply

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3, 3))]
        for layer_index in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden), nn.LayerNorm()]
        layers += [ConcatELU(), nn.Conv(self.c_out, kernel_size=(3, 3), kernel_init=nn.initializers.zeros)]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)


# ------------------------------- Multiscale -------------------------------

class SqueezeFlow(nn.Module):

    def __call__(self, z, ldj, rng, reverse=False):
        B, H, W, C = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, H // 2, 2, W // 2, 2, C)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H // 2, W // 2, 4 * C)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, H, W, 2, 2, C // 4)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H * 2, W * 2, C // 4)
        return z, ldj, rng


class SplitFlow(nn.Module):

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, z_split = z.split(2, axis=-1)
            ldj += jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
        else:
            z_split = random.normal(rng, z.shape)
            z = np.concatenate([z, z_split], axis=-1)
            ldj -= jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
        return z, ldj, rng

# ------------------------------- Masks (abstract class) -------------------------------


class FlowMasks:
    def create_checkerboard_mask(h, w, invert=False):
        x, y = np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        mask = np.fmod(xx + yy, 2)
        mask = mask.astype(np.float32).reshape(1, h, w, 1)
        if invert:
            mask = 1 - mask
        return mask

    def create_channel_mask(c_in, invert=False):
        mask = np.concatenate([np.ones((c_in // 2,), dtype=np.float32),
                               np.zeros((c_in - c_in // 2,), dtype=np.float32)])
        mask = mask.reshape(1, 1, 1, c_in)
        if invert:
            mask = 1 - mask
        return mask
