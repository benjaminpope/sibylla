# JAX
import jax
import jax.numpy as np
from jax import random
import layers
from flows import ImageFlow, FlowFactory

import os
from TrainerModule import TrainerModule
from sibylla.Norm_Flows.data_loaders import ImageDataLoader
from typing import Sequence

from flax import linen as nn


class mySplitFlow(nn.Module):

    def __call__(self, z, z_split, ldj, rng, reverse=False):
        if not reverse:
            z, z_split = z.split(2, axis=-1)
            ldj += jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
        else:
            z = np.concatenate([z, z_split], axis=-1)
            ldj -= jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
            z_split = []
        return z, z_split, ldj, rng


class myImageFlow(nn.Module):
    flows: Sequence[nn.Module]  # A list of flows (each a nn.Module) that should be applied on the images.
    import_samples: int = 8  # Number of importance samples to use during testing (see explanation below).

    def __call__(self, x, rng=None, testing=False):
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

    def encode(self, imgs, rng=None):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, z_split, ldj = imgs, [], np.zeros(imgs.shape[0])
        for flow in self.flows:
            z, z_split, ldj, rng = flow(z, z_split, ldj, rng, reverse=False)
        z = np.concatenate([z, z_split], axis=-1)
        return z, ldj, rng

    def _get_likelihood(self, imgs, rng=None, return_ll=False):
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
        bpd = nll * np.log2(np.exp(1)) / np.prod(np.array(imgs.shape[1:]))
        return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            rng, normal_rng = random.split(rng)
            z = random.normal(normal_rng, shape=img_shape)
            z_split = random.normal(normal_rng, shape=img_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        ldj = np.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, z_split, ldj, rng = flow(z, z_split, ldj, rng, reverse=True)
        return z, rng


class myCouplingLayer(nn.Module):
    network: nn.Module  # NN to use in the flow for predicting mu and sigma
    mask: np.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in: int  # Number of input channels

    def setup(self):
        self.scaling_factor = self.param('scaling_factor', nn.initializers.zeros, (self.c_in,))

    def __call__(self, z, z_split, ldj, rng, reverse=False, orig_img=None):
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

        return z, z_split, ldj, rng


class mySqueezeFlow(nn.Module):

    def __call__(self, z, z_split, ldj, rng, reverse=False):
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
        return z, z_split, ldj, rng


class myFlowFactory:
    def create_split_flow():
        flow_layers = []
        # vardeq_layers = [layers.CouplingLayer(network=layers.GatedConvNet(c_out=2, c_hidden=16),
        #                                       mask=layers.FlowMasks.create_checkerboard_mask(h=28,
        #                                                                                      w=28,
        #                                                                                      invert=(i % 2 == 1)),
        #                                       c_in=1) for i in range(4)]
        # flow_layers += [layers.VariationalDequantization(var_flows=vardeq_layers)]

        flow_layers += [myCouplingLayer(network=layers.GatedConvNet(c_out=2, c_hidden=32),
                                        mask=layers.FlowMasks.create_checkerboard_mask(h=28,
                                                                                       w=28,
                                                                                       invert=(i % 2 == 1)),
                                        c_in=1) for i in range(2)]
        flow_layers += [mySqueezeFlow(),
                        mySqueezeFlow(),
                        mySplitFlow()]
        flow_model = myImageFlow(flow_layers)
        return flow_model


if __name__ == "__main__":
    checkpoint_path = "./saved_models/simple_example"
    model_name = "MNISTFlow_multiscale"

    # Use pretrained model
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial11/"
    # Files to download
    pretrained_files = ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt",
                        "MNISTFlow_simple_results.json", "MNISTFlow_vardeq_results.json",
                        "MNISTFlow_multiscale_results.json"]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(checkpoint_path, exist_ok=True)

    print("Checking pretrained files...", end='')
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(checkpoint_path, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please contact the author with the full output including the"
                      "following error:\n", e)
    print("Done!")

    print("Loading dataset...", end='')
    train_set, val_set, test_set = ImageDataLoader.load_MNIST()
    train_exmp_loader, train_data_loader, \
        val_loader, test_loader = ImageDataLoader.generate_data_loaders(train_set, val_set, test_set)
    print("Done!")

    # show_imgs(np.stack([train_set[i][0] for i in range(8)], axis=0))

    exmp_imgs = next(iter(train_exmp_loader))[0]
    img = exmp_imgs[0:1]
    rng = jax.random.PRNGKey(42)
    
    print("Creating flow...", end='')
    flow_dict = {"simple": {}, "vardeq": {}, "multiscale": {}}
    flow = FlowFactory.create_split_flow()
    exmp_imgs = next(iter(train_exmp_loader))[0]
    rng, init_rng, flow_rng = jax.random.split(rng, 3)
    params = flow.init(init_rng, exmp_imgs, flow_rng)['params']
    print("Done!")

    model = flow.bind({'params': params})
    encoding = model.encode(img, rng)[0]

    print(f"Encoded size from UvA model {encoding.shape}, size {encoding.size}. Img: {img.shape}, {img.size}")
    
    # now do custom model
    
    myflow = myFlowFactory.create_split_flow()
    # trainer = TrainerModule(model_name, train_exmp_loader, train_data_loader, checkpoint_path, flow)
    # model, _ = trainer.train_flow()
    rng, init_rng, flow_rng = jax.random.split(rng, 3)
    params = myflow.init(init_rng, exmp_imgs, flow_rng)['params']
    
    mymodel = myflow.bind({'params': params})
    myencoding = mymodel.encode(img, rng)[0]
    print(f"Encoded size from custom model {myencoding.shape}, size {myencoding.size}. Img: {img.shape}, {img.size}")
    
    
    # now testing sampling
    sampled_imgs, _ = model.sample([16, 7, 7, 8], rng)
    print(f"Sampled images from UvA model have shape {sampled_imgs.shape}")
    
    sampled_imgs, _ = mymodel.sample([16, 7, 7, 8], rng)
    print(f"Sampled images from custom model have shape {sampled_imgs.shape}")