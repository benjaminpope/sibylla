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
    def __init__(self, model, save_figs=False, save_pth=None):
        self.model = model

        if save_figs is True and save_pth is None:
            raise ValueError("empty save path not permitted")

        self.save_figs = save_figs
        self.save_pth = save_pth

        self.save_pth = os.path.join(save_pth)
        if not os.path.exists(self.save_pth):
            os.mkdir(self.save_pth)

    def random_sample(self):
        sample_rng = random.PRNGKey(44)
        samples, _ = self.model.model_bd.sample(img_shape=[16, 7, 7, 8], rng=sample_rng)
        self.show_imgs(samples, title='Random samples from flow')

    def show_imgs(self, imgs, savename='', **kwargs):
        ModelEvaluator.plt_imgs(imgs, kwargs)
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
