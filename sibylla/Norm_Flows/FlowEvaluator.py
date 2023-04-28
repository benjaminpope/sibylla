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
        if self.save_plots:
            assert save_name is not None, "must provide save name"
            plt.savefig(os.path.join(self.save_path, f"{save_name}{self.save_extension}"))
        if self.show_plots:
            plt.show()

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

            log_likelihoods = FlowEvaluator.remove_infs(log_likelihoods, ds_name)
            plt.hist(log_likelihoods, label=ds_name, **hist_opts)
        
        # compute for noise images
        imgs = jax.random.uniform(next(prng_seq), imgs.shape)
        log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(self.params, imgs[:n_imgs])
        log_likelihoods = FlowEvaluator.remove_infs(log_likelihoods, 'noise data')
        plt.hist(log_likelihoods, label='noise', **hist_opts)

        # inverted training images
        imgs = ImageDataset.normalize_dequant_data(next(dict_of_ds['train']), next(prng_seq))
        imgs = 1. - imgs
        log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(self.params, imgs[:n_imgs])
        log_likelihoods = FlowEvaluator.remove_infs(log_likelihoods, 'inverted data')
        plt.hist(log_likelihoods, label='inverted', **hist_opts)

        plt.legend()
        self.finish_plot('encoded_hist')

    def remove_infs(arr, msg=None):
        if jnp.isinf(arr).any():
            if msg is not None:
                logging.warn(f"Removing infs from {msg}")
            arr = arr[jnp.isfinite(arr)]
        return arr

    def display_small_scale_var(self, base_img, n_imgs = 5):
        """
        for the multiscale model, run the generative direction several times, varying only the elements of the input that are masked
        in the ignorance layers
        """
        # encode the base image
        base_img_z = NormFlow.get_inverse_model(self.config).apply(self.params, base_img)

        event_shape = self.config.data_shape

        mask = jnp.arange(0, np.prod(np.array(event_shape))) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)
        ignorance_mask = mask


        forward_model = NormFlow.get_forward_model(self.config)

        plt.figure()
        imshow_args = {
             'vmin' : 0, 
             'vmax' : 1,
             'cmap' : 'gray'
        }

        plt.subplot(1,n_imgs+1, 1)
        plt.imshow(forward_model.apply(self.params, base_img_z), **imshow_args)
        plt.title('Starting img')
        
        base_distribution = NormFlow.get_base_distribution(self.config).apply(self.params)
        new_random_draws = base_distribution._sample_n(0, n_imgs)
        for i in range(n_imgs):
            new_z = np.where(ignorance_mask, base_img_z, new_random_draws[i])
            fwd = forward_model.apply(self.params, new_z)
            plt.subplot(1,n_imgs+1, i+2)
            plt.imshow(fwd, **imshow_args)

        self.finish_plot('display_multiscale')

    def display_fwd_inv(self, img, save_name = ''):
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
        self.finish_plot('display_fwd_inv' + save_name)

    def bits_per_dim(self, img_batch):
        """
        return the image
        """
        log_prob = NormFlow.get_log_prob(self.config)

        return np.mean(log_prob.apply(self.params, img_batch)) * np.log2(np.exp(1)) / np.prod(img_batch.shape[1:])

    def get_num_params(self):
        return sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(self.params)])

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

    
