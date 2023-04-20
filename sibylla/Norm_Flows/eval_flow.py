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
import different_mask_flow_uniform_config as different_mask_flow_uniform_config
import random_mask_flow_uniform_config as random_mask_flow_uniform_config
from ImageDataset import ImageDataset
from NormFlow import NormFlow
from FlowEvaluator import FlowEvaluator

jax.random.PRNGKey(4)

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_integer('version', -1, 'which version of the model to use')
flags.DEFINE_enum('flow_model', 'uniform_base_flow',
                  ['uniform_base_flow','different_mask_flow','random_mask_flow'], 'Flow to eval')
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
    elif FLAGS.flow_model == "different_mask_flow":
        config = different_mask_flow_uniform_config.get_config(FLAGS.dataset)
    elif FLAGS.flow_model == "random_mask_flow":
        config = random_mask_flow_uniform_config.get_config(FLAGS.dataset)
    else:
        raise KeyError(f'{FLAGS.flow_model} is not implemented!')



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


    train_ds, eval_ds = ImageDataset.get_train_test_iterators('mnist', config.train.batch_size, config.eval.batch_size)
    etrain_ds, _ = ImageDataset.get_train_test_iterators('emnist', config.train.batch_size, config.eval.batch_size)

    prng_seq = hk.PRNGSequence(42)

    evaluator = FlowEvaluator(config, version=FLAGS.version, show_plots=True, save_plots=True)

    img = ImageDataset.normalize_dequant_data(next(eval_ds), next(prng_seq))[0]
    evaluator.display_fwd_inv(img)
    
    dict_of_ds = {
        'train' : train_ds, 
        'eval' : eval_ds,
        'emnist' : etrain_ds
    }
    evaluator.show_encoded_hist(dict_of_ds, prng_seq, n_imgs=256)


    exit()

    # show_samples(params, prng_seq)
    imgs = ImageDataset.normalize_dequant_data(next(eval_ds), next(prng_seq))[0:10]
    time_model(params, imgs, n_runs=5)
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

    # noise = jax.random.uniform(next(prng_seq), img.shape)
    # display_fwd_inv(params, noise)


if __name__ == '__main__':
    app.run(main)
