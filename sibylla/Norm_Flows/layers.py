"""
layers for normalising flows
modified from
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial11/NF_image_modeling.html
"""

# Standard libraries
from typing import Sequence

import numpy as np

# JAX
import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn


# ------------------------------- Base flow -------------------------------

class ImageFlow(nn.Module):
    flows: Sequence[nn.Module]  # A list of flows (each a nn.Module) that should be applied on the images.
    import_samples: int = 8  # Number of importance samples to use during testing (see explanation below).

    def __call__(self, x, rng, testing=False):
        if not testing:
            bpd, rng = self._get_likelihood(x, rng)
        else:
            # Perform importance sampling during testing => estimate likelihood M times for each image
            img_ll, rng = self._get_likelihood(x.repeat(self.import_samples, 0), rng, return_ll=True)
            img_ll = img_ll.reshape(-1, self.import_samples)

            # To average the probabilities, we need to go from log-space to exp, and back to log.
            # Logsumexp provides us a stable implementation for this
            img_ll = jax.nn.logsumexp(img_ll, axis=-1) - np.log(self.import_samples)

            # Calculate final bpd
            bpd = -img_ll * np.log2(np.exp(1)) / np.prod(x.shape[1:])
            bpd = bpd.mean()
        return bpd, rng

    def encode(self, imgs, rng):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, jnp.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj, rng = flow(z, ldj, rng, reverse=False)
        return z, ldj, rng

    def _get_likelihood(self, imgs, rng, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj, rng = self.encode(imgs, rng)
        log_pz = jax.scipy.stats.norm.logpdf(z).sum(axis=(1, 2, 3))
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            rng, normal_rng = random.split(rng)
            z = random.normal(normal_rng, shape=img_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        ldj = jnp.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj, rng = flow(z, ldj, rng, reverse=True)
        return z, rng


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
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = jnp.floor(z)
            z = jax.lax.clamp(min=0.0, x=z, max=self.quants - 1.0).astype(jnp.int32)
        return z, ldj, rng

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z - 2 * jax.nn.softplus(-z)).sum(axis=[1, 2, 3])
            z = nn.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-jnp.log(z) - jnp.log(1 - z)).sum(axis=[1, 2, 3])
            z = jnp.log(z) - jnp.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj, rng):
        # Transform discrete values to continuous volumes
        z = z.astype(jnp.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj, rng


class VariationalDequantization(Dequantization):
    var_flows: Sequence[nn.Module] = None  # A list of flow transformations to use for modeling q(u|x)

    def dequant(self, z, ldj, rng):
        z = z.astype(jnp.float32)
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
        ldj -= np.log(256.0) * np.prod(z.shape[1:])
        return z, ldj, rng


# ------------------------------- Standard layers -------------------------------


class CouplingLayer(nn.Module):
    network: nn.Module  # NN to use in the flow for predicting mu and sigma
    mask: np.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
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
            nn_out = self.network(jnp.concatenate([z_in, orig_img], axis=-1))
        s, t = nn_out.split(2, axis=-1)

        # Stabilize scaling output
        s_fac = jnp.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1, 2, 3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1, 2, 3])

        return z, ldj, rng


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


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
