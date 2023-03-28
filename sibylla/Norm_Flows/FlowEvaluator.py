from NormFlow import NormFlow
import torchvision.utils
import math
import torch
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import os

from absl import logging

from ImageDataset import ImageDataset
from ModelStorage import ModelStorage


class FlowEvaluator:
    def __init__(self, config, version=-1, show_plots=True, save_plots=False, save_extension='.pdf') -> None:
        assert not (save_plots == False and show_plots == False), "Why are you running this without plots??"

        self.config = config

        self.show_plots = show_plots
        self.save_plots = save_plots

        # load params
        pth, self.version = ModelStorage.get_model_path(config, version)
        self.params = ModelStorage.load_model(pth)

        logging.info(f"Evaluating the {config.model_name} model, version {version}(={self.version})")

        if save_plots:
            self.save_path = self.get_results_path()
            self.save_extension = save_extension

        self.create_model = NormFlow.get_create_model(config)

    def finish_plot(self, save_name=None):
        """
        Function to call when done with a plot, it will save it or show it depending on the settings
        """
        if self.show_plots:
            plt.show()
        if self.save_plots:
            assert save_name is not None, "must provide save name"
            plt.savefig(os.path.join(self.save_path, f"{save_name}{self.save_extension}"))

    def get_results_path(self):
        """ Create a folder to save the model in and return the path"""
        model_path = os.path.join('results', self.config.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if self.version < 0:
            files = os.listdir(model_path)
            files.sort()
            target_version = files[self.version]
        else:
            target_version = f'version_{self.version}'
        version_path = os.path.join(model_path, target_version)
        if not os.path.exists(version_path):
            os.makedirs(version_path)
        return version_path

    def show_encoded_hist(self, dict_of_ds, prng_seq, n_imgs=None):
        """
        Display a histogram showing how the encoded space looks
            - dict_of_ds: a dictionary of dataset generators e.g. {'train': train_ds, 'e_mnist': emnist_ds} 
            - params: model parameters
            - prng_seq: rng sequence
            - x_scale: "distance" if the histogram should show the distance from the origin,
                       "log_likelihood" if the histogram should show the likelihood of the encoded image
            - norm_x_scale: if the x scale should be normalised independent of image dimension

        """
        hist_opts = {'lw' : 0.1, 'alpha' : 0.7, 'edgecolor' : 'k'}

        log_prob = NormFlow.get_log_prob(self.config)

        if n_imgs is None:
            n_imgs = imgs.shape[0]

        for ds_name, ds in dict_of_ds.items():
            imgs = ImageDataset.normalize_dequant_data(next(ds), next(prng_seq))
            log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(self.params, imgs[:n_imgs])

            plt.hist(log_likelihoods, label=ds_name, **hist_opts)
        
        # compute for noise images
        imgs = jax.random.uniform(next(prng_seq), imgs.shape)
        log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(self.params, imgs[:n_imgs])
        plt.hist(log_likelihoods, label='noise', **hist_opts)

        # inverted training images
        # imgs = ImageDataset.normalize_dequant_data(next(dict_of_ds['train']), next(prng_seq))
        # imgs = 1. - imgs
        # log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(self.params, imgs[:n_imgs])
        # plt.hist(log_likelihoods, label='inverted', **hist_opts)

        plt.legend()
        self.finish_plot('encoded_hist')

    def display_fwd_inv(self, img):
        forward_model = NormFlow.get_forward_model(self.config)
        inverse_model = NormFlow.get_inverse_model(self.config)

        fwd = forward_model.apply(self.params, img)
        inv = inverse_model.apply(self.params, img)

        imshow_args = {
             'vmin' : 0, 
             'vmax' : 1,
             'cmap' : 'gray'
        }

        logging.info(f"Norms of: img {jnp.linalg.norm(img)}, fwd {jnp.linalg.norm(fwd)}, inv {jnp.linalg.norm(inv)}")
        plt.subplot(131)
        plt.imshow(img, **imshow_args)
        plt.title('Input image')
        plt.subplot(132)
        plt.imshow(fwd, **imshow_args)
        plt.title('Forward( image )')
        plt.subplot(133)
        plt.imshow(inv, **imshow_args)
        plt.title('Inverse( image )')
        self.finish_plot('display_fwd_inv')

    def show_img_grid(imgs, row_size=4):
        """
        class helper method to show images in a compact grid
        """
        num_imgs = imgs.shape[0]
        nrow = min(num_imgs, row_size)
        ncol = int(math.ceil(num_imgs / nrow))
        imgs_torch = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)
        imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow)
        np_imgs = imgs.cpu().numpy()
        plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
        plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
        plt.axis('off')
        plt.show()

    
