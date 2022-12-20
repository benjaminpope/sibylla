import math
import torchvision
import torch
import jax
import jax.numpy as np
import numpy as onp
import os

import jax.random as random
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, model, training_imgs, save_figs=False, save_pth=None):
        self.model = model

        self.training_imgs = training_imgs

        if save_figs is True and save_pth is None:
            raise ValueError("empty save path not permitted")

        self.save_figs = save_figs
        if save_figs is True:
            self.save_pth = save_pth

            self.save_pth = os.path.join(save_pth)
            if not os.path.exists(self.save_pth):
                os.mkdir(self.save_pth)

    def random_sample(self):
        sample_rng = random.PRNGKey(44)
        samples, _ = self.model.sample(img_shape=[16, 7, 7, 8], rng=sample_rng)
        self.show_imgs(samples, savename='random_sample.png',title='Random samples from flow')

    def standard_interp(self, save=True, n_imgs=2):
        # interp between some samples
        n_step = 8
        rng = jax.random.PRNGKey(42)
        for i in range(n_imgs):
            interp_imgs = self._interpolate(rng, self.training_imgs[2 * i], self.training_imgs[2 * i + 1], n_step)
            self.show_imgs(interp_imgs, savename=f'standard_interp_{i}.png', row_size=n_step)

    def _interpolate(self, rng, img1, img2, num_steps=8):
        """
        Inputs:
            img1, img2 - Image tensors of shape [1, 28, 28]. Images between which should be interpolated.
            num_steps - Number of interpolation steps. 8 interpolation steps mean 6 intermediate pictures 
            besides img1 and img2
        """
        imgs = np.stack([img1, img2], axis=0)
        z, _, rng = self.model.encode(imgs, rng)
        alpha = np.linspace(0, 1, num=num_steps).reshape(-1, 1, 1, 1)
        interpolations = z[0:1] * alpha + z[1:2] * (1 - alpha)
        interp_imgs, _ = self.model.sample(interpolations.shape[:1] + imgs.shape[1:], rng=rng, z_init=interpolations)
        return interp_imgs

    def show_imgs(self, imgs, savename='', **kwargs):
        ModelEvaluator.plt_imgs(imgs, **kwargs)
        if self.save_figs:
            plt.savefig(os.path.join(self.save_pth, savename))
        else:
            plt.show()
            plt.close()

    def plt_imgs(imgs, title=None, row_size=4):
        # Form a grid of pictures (we use max. 8 columns)
        imgs = np.copy(jax.device_get(imgs))
        num_imgs = imgs.shape[0]
        is_int = (imgs.dtype == np.int32)
        nrow = min(num_imgs, row_size)
        ncol = int(math.ceil(num_imgs / nrow))
        imgs_torch = torch.from_numpy(onp.array(imgs)).permute(0, 3, 1, 2)
        imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow, pad_value=128 if is_int else 0.5)
        np_imgs = imgs.cpu().numpy()
        # Plot the grid
        plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
        plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation='nearest')
        plt.axis('off')
        if title is not None:
            plt.title(title)
