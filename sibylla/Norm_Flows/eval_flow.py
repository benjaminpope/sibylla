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
import tensorflow_datasets as tfds
from ModelStorage import ModelStorage
import timeit

import matplotlib.pyplot as plt
import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config
from ImageDataset import ImageDataset
from NormFlow import NormFlow

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

    inverse_model = NormFlow.get_inverse_model(config)
    forward_model = NormFlow.get_forward_model(config)

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

    log_prob = NormFlow.get_log_prob(config)
    
    @jax.jit
    def fast_log_prob(params, imgs):
        return log_prob.apply(params, imgs)

    def time_model(params, imgs, n_runs=10):
        # make sure jited 
        p = fast_log_prob(params, imgs)

        t = timeit.timeit(lambda: fast_log_prob(params, imgs), number = n_runs)
        logging.info(f"{t/n_runs:.3e} seconds for {imgs.shape[0]} images (avg over {n_runs} runs)")
        # t0 = time.clock()
        # for _ in range(n_runs):
        #     x = log_prob.apply(params, img)
        # t1 = time.clock()
        # logging.info(f"{(t1-t0)/n_runs:.3e} seconds for {img.shape[0]} images (avg over )")

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

    def show_encoded_hist(dict_of_ds, params, prng_seq, n_imgs=None):
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

        if n_imgs is None:
            n_imgs = imgs.shape[0]

        for ds_name, ds in dict_of_ds.items():
            imgs = ImageDataset.normalize_dequant_data(next(ds), next(prng_seq))
            log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(params, imgs[:n_imgs])

            plt.hist(log_likelihoods, label=ds_name, **hist_opts)
        
        # compute for noise images
        imgs = jax.random.uniform(next(prng_seq), imgs.shape)
        log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(params, imgs[:n_imgs])
        plt.hist(log_likelihoods, label='noise', **hist_opts)

        # inverted training images
        # imgs = ImageDataset.normalize_dequant_data(next(dict_of_ds['train']), next(prng_seq))
        # imgs = 1. - imgs
        # log_likelihoods = jax.vmap(log_prob.apply, in_axes=(None, 0))(params, imgs[:n_imgs])
        # plt.hist(log_likelihoods, label='inverted', **hist_opts)

        plt.legend()
        plt.show()

    train_ds, eval_ds = ImageDataset.get_train_test_iterators('mnist', config.train.batch_size, config.eval.batch_size)
    etrain_ds, _ = ImageDataset.get_train_test_iterators('emnist', config.train.batch_size, config.eval.batch_size)

    # load params
    params = ModelStorage.load_model(save_path)

    prng_seq = hk.PRNGSequence(42)

    # show_samples(params, prng_seq)
    imgs = ImageDataset.normalize_dequant_data(next(eval_ds), next(prng_seq))[0:10]
    time_model(params, imgs, n_runs=5)
    exit()
    # dict_of_ds = {
    #     'train' : train_ds, 
    #     'eval' : eval_ds,
    #     'emnist' : etrain_ds
    # }
    # show_encoded_hist(dict_of_ds, params, prng_seq,n_imgs=2)
    # exit()
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
