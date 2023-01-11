
import numpy as onp
from jax import random

# JAX
import jax
import jax.numpy as np

from typing import Sequence
from flax import linen as nn
import layers


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
            img_ll = jax.nn.logsumexp(img_ll, axis=-1) - onp.log(self.import_samples)

            # Calculate final bpd
            bpd = -img_ll * onp.log2(onp.exp(1)) / onp.prod(x.shape[1:])
            bpd = bpd.mean()
        return bpd, rng

    def encode(self, imgs, rng):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, np.zeros(imgs.shape[0])
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
        bpd = nll * onp.log2(onp.exp(1)) / onp.prod(imgs.shape[1:])
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
        ldj = np.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj, rng = flow(z, ldj, rng, reverse=True)
        return z, rng


class FlowFactory:
    def create_multiscale_flow():
        flow_layers = []
        vardeq_layers = [layers.CouplingLayer(network=layers.GatedConvNet(c_out=2, c_hidden=16),
                                              mask=layers.FlowMasks.create_checkerboard_mask(h=28,
                                                                                             w=28,
                                                                                             invert=(i % 2 == 1)),
                                              c_in=1) for i in range(4)]
        flow_layers += [layers.VariationalDequantization(var_flows=vardeq_layers)]

        flow_layers += [layers.CouplingLayer(network=layers.GatedConvNet(c_out=2, c_hidden=32),
                                             mask=layers.FlowMasks.create_checkerboard_mask(h=28,
                                                                                            w=28,
                                                                                            invert=(i % 2 == 1)),
                                             c_in=1) for i in range(2)]
        flow_layers += [layers.SqueezeFlow()]
        for i in range(2):
            flow_layers += [layers.CouplingLayer(network=layers.GatedConvNet(c_out=8, c_hidden=48),
                                                 mask=layers.FlowMasks.create_channel_mask(c_in=4,
                                                                                           invert=(i % 2 == 1)),
                                                 c_in=4)]
        flow_layers += [layers.SplitFlow(),
                        layers.SqueezeFlow()]
        for i in range(4):
            flow_layers += [layers.CouplingLayer(network=layers.GatedConvNet(c_out=16, c_hidden=64),
                                                 mask=layers.FlowMasks.create_channel_mask(c_in=8,
                                                                                           invert=(i % 2 == 1)),
                                                 c_in=8)]
        flow_model = ImageFlow(flow_layers)
        return flow_model

    def create_split_flow():
        flow_layers = []
        vardeq_layers = [layers.CouplingLayer(network=layers.GatedConvNet(c_out=2, c_hidden=16),
                                              mask=layers.FlowMasks.create_checkerboard_mask(h=28,
                                                                                             w=28,
                                                                                             invert=(i % 2 == 1)),
                                              c_in=1) for i in range(4)]
        flow_layers += [layers.VariationalDequantization(var_flows=vardeq_layers)]

        flow_layers += [layers.SqueezeFlow(),
                        layers.SqueezeFlow(),
                        layers.SplitFlow()]
        flow_model = ImageFlow(flow_layers)
        return flow_model
