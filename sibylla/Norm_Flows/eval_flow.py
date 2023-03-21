"""Evaluates a trained flow on data"""


from typing import Union, Any, Iterator, Mapping, Optional

from absl import app
from absl import flags
from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from ModelStorage import ModelStorage
import torchvision.utils
import math
import torch

import matplotlib.pyplot as plt
import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config
from ImageDataset import ImageDataset

jax.random.PRNGKey(4)

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_integer('version', -1, 'which version of the model to use')
flags.DEFINE_enum('flow_model', 'uniform_base_flow',
                  ['uniform_base_flow'], 'Flow to train')
flags.DEFINE_enum('dataset', 'MNIST',
                  ['MNIST'], 'Dataset to train')

FLAGS = flags.FLAGS

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any



def show_img_grid(imgs, row_size=4):
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


def main(_):
    if FLAGS.flow_model == "uniform_base_flow":
        config = uniform_base_flow_config.get_config(FLAGS.dataset)
    else:
        raise KeyError(f'{FLAGS.flow_model} is not implemented!')

    logging.info(f"evaluating the {config.model_name} model, version {FLAGS.version}")
    save_path = ModelStorage.get_model_path(config, version=FLAGS.version)


    def create_model():
        return config.model['constructor'](
            **config.model['kwargs'])

    @hk.without_apply_rng
    @hk.transform
    def inverse_model(data):
        model = create_model()
        return model.bijector.inverse(data)

    @hk.without_apply_rng
    @hk.transform
    def forward_model(data):
        model = create_model()
        return model.bijector.forward(data)

    @hk.without_apply_rng
    @hk.transform
    def sample_from_base_distribution(n_samples, prng_key):
        model = create_model()
        return model.distribution._sample_n(prng_key, n_samples)

    @hk.without_apply_rng
    @hk.transform
    def get_base_distribution():
        model = create_model()
        return model.distribution

    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: Array) -> Array:
        model = create_model()
        return model.log_prob(data)
    
    def display_fwd_inv(params, img):
        fwd = forward_model.apply(params, img)
        inv = inverse_model.apply(params, img)

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
        plt.show()

    def encode_different_data(dict_of_ds, params, prng_seq):
        encoded = {}
        for key, ds in dict_of_ds.items():
            imgs = ImageDataset.normalize_dequant_data(next(ds), next(prng_seq))
            encoded[key] = inverse_model.apply(params, imgs)
        noise = jax.random.uniform(next(prng_seq), imgs.shape)
        train_imgs = ImageDataset.normalize_dequant_data(next(dict_of_ds['train']), next(prng_seq))
        inverted = 1 - train_imgs

        encoded['noise'] = inverse_model.apply(params, noise)
        encoded['inverted'] = inverse_model.apply(params, inverted)

        return encoded

    def get_samples(n_samples, params, prng_seq, draw_from='model'):
        base_samples = sample_from_base_distribution.apply(params, n_samples, next(prng_seq))
        if draw_from == 'base':
            return base_samples
        elif draw_from == 'model':
            return forward_model.apply(params, base_samples)
        elif draw_from == 'uniform':
            return jax.random.uniform(next(prng_seq), base_samples.shape)
        elif draw_from == 'model_uniform':
            samples = jax.random.uniform(next(prng_seq), base_samples.shape)
            return forward_model.apply(params, samples)

    def show_encoded_hist(dict_of_ds, params, prng_seq, x_scale="distance", norm_x_scale=True):
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

        for ds_name, ds in dict_of_ds.items():
            imgs = ImageDataset.normalize_dequant_data(next(ds), next(prng_seq))
            n_imgs = imgs.shape[0]
            n_imgs = 20
            log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(params, imgs[:n_imgs])

            plt.hist(log_likelihoods, label=ds_name, **hist_opts)
        
        # TODO: make noise and inverted easier to use here...
        imgs = jax.random.uniform(next(prng_seq), imgs.shape)
        log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(params, imgs[:n_imgs])
        plt.hist(log_likelihoods, label='noise', **hist_opts)

        plt.legend()
        plt.show()

    train_ds, eval_ds = ImageDataset.get_train_test_iterators('mnist', config.train.batch_size, config.eval.batch_size)
    # etrain_ds, _ = ImageDataset.get_train_test_iterators('emnist', config.train.batch_size, config.eval.batch_size)

    # load params
    params = ModelStorage.load_model(save_path)

    prng_seq = hk.PRNGSequence(42)

    # show_samples(params, prng_seq)

    # show_encoded_hist(train_ds, eval_ds, params, prng_seq)4
    dict_of_ds = {
        'train' : train_ds, 
        'eval' : eval_ds
    }
    show_encoded_hist(dict_of_ds, params, prng_seq, x_scale="log_likelihood")
    exit()
    # train_imgs = prepare_data(next(train_ds), next(prng_seq))
    # show_img_grid(train_imgs[0:8,:,:,:])

    # sampled_imgs = get_samples(8, params, prng_seq, draw_from='model')
    # sampled_imgs = get_samples(8, params, prng_seq, draw_from='model_uniform')
    # show_img_grid(sampled_imgs)

    img = ImageDataset.normalize_dequant_data(next(eval_ds), next(prng_seq))[0]
    display_fwd_inv(params, img)
    # noise = jax.random.uniform(next(prng_seq), img.shape)
    # display_fwd_inv(params, noise)


if __name__ == '__main__':
    app.run(main)
