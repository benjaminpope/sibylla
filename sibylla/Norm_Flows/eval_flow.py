"""Evaluates a trained flow on data"""


from typing import Union, Any, Iterator, Mapping, Optional

from absl import app
from absl import flags
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
import simple_flow_config
import simple_flow_config_v2

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('system', 'simple_MNIST',
                  ['simple_MNIST'], 'Experiment and dataset to train')
flags.DEFINE_integer('version', -1, 'which version of the model to use')

FLAGS = flags.FLAGS

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
    data = batch["image"].astype(np.float32)
    if prng_key is not None:
        # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
        data += jax.random.uniform(prng_key, data.shape)
    return data / 256.  # Normalize pixel values from [0, 256) to [0, 1).


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    ds = tfds.load("mnist", split=split, shuffle_files=False)
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))

def show_img_grid(imgs, row_size=4):
    num_imgs = imgs.shape[0]
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs_torch = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)
    imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow)
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)))
    plt.axis('off')
    plt.show()


def main(_):
    system = FLAGS.system
    if True:
        # config = simple_flow_config.get_config('MNIST')
        config = simple_flow_config_v2.get_config('MNIST')
    else:
        raise KeyError(system)

    print(f"evaluationg the {config.model_name} model, version {FLAGS.version}")
    save_path = ModelStorage.get_model_path(config, version=FLAGS.version)

    optimizer = optax.adam(config.train.learning_rate)
    if config.train.max_gradient_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.max_gradient_norm), optimizer)

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
    
    def display_fwd_inv(params, img):
        fwd = forward_model.apply(params, img)
        inv = inverse_model.apply(params, img)

        print(f"Norms of: img {jnp.linalg.norm(img)}, fwd {jnp.linalg.norm(fwd)}, inv {jnp.linalg.norm(inv)}")
        plt.subplot(131)
        plt.imshow(img, vmin=0, vmax=1)
        plt.title('Input image')
        plt.subplot(132)
        plt.imshow(fwd, vmin=0, vmax=1)
        plt.title('Forward( image )')
        plt.subplot(133)
        plt.imshow(inv, vmin=0, vmax=1)
        plt.title('Inverse( image )')
        plt.show()
    
    def encode_different_data(train_ds, eval_ds, params, prng_seq):
        train_imgs = prepare_data(next(train_ds), next(prng_seq))
        eval_imgs = prepare_data(next(eval_ds), next(prng_seq))
        noise = jax.random.uniform(next(prng_seq), train_imgs.shape)
        inverted = 1 - train_imgs
        encoded = {}
        
        encoded['train'] = inverse_model.apply(params, train_imgs)
        encoded['eval'] = inverse_model.apply(params, eval_imgs)
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
    
    def show_encoded_hist(train_ds, eval_ds, params, prng_seq, x_scale="distance", norm_x_scale=True):
        """ 
        Display a histogram showing how the encoded space looks
            - train_ds: training dataset
            - eval_ds: evaluation dataset
            - params: model parameters
            - prng_seq: rng sequence
            - x_scale: "distance" if the histogram should show the distance from the origin,
                       "log_likelihood" if the histogram should show the likelihood of the encoded image
            - norm_x_scale: if the x scale should be normalised independent of 
        
        """
        hist_opts = {'lw' : 0.1, 'alpha' : 0.7, 'edgecolor' : 'k'}
        encoded = encode_different_data(train_ds, eval_ds, params, prng_seq)
        
        for dataset, encoded_imgs in encoded.items():
            if x_scale == "distance":
                distances = np.zeros((encoded_imgs.shape[0],))
                for idx in range(encoded_imgs.shape[0]):
                    distances[idx] = jnp.linalg.norm(encoded_imgs[idx,:,:,:])
                plt.hist(distances, label=dataset, **hist_opts)
            elif x_scale == "log_likelihood":
                base = get_base_distribution.apply(params)
                log_likelihoods = np.zeros((encoded_imgs.shape[0],))
                for idx in range(encoded_imgs.shape[0]):
                    log_likelihoods[idx] = base.log_prob(encoded_imgs[idx,:,:,:])
                plt.hist(log_likelihoods, label=dataset, **hist_opts)
            else:
                raise ValueError(f"{x_scale} is not a valid entry for x_scale")

        plt.legend()
        plt.show()
    
    train_ds = load_dataset(tfds.Split.TRAIN, config.eval.batch_size)
    eval_ds = load_dataset(tfds.Split.TEST, config.eval.batch_size)

    # load params
    params = ModelStorage.load_model(save_path)

    prng_seq = hk.PRNGSequence(42)

    # show_samples(params, prng_seq)

    # show_encoded_hist(train_ds, eval_ds, params, prng_seq)
    # show_encoded_hist(train_ds, eval_ds, params, prng_seq, x_scale="log_likelihood")

    # train_imgs = prepare_data(next(train_ds), next(prng_seq))
    # show_img_grid(train_imgs[0:8,:,:,:])

    sampled_imgs = get_samples(8, params, prng_seq, draw_from='model')
    show_img_grid(sampled_imgs)

    # img = prepare_data(next(eval_ds), next(prng_seq))[0]
    # display_fwd_inv(params, img)
    # noise = jax.random.uniform(next(prng_seq), img.shape)
    # display_fwd_inv(params, noise)


if __name__ == '__main__':
    app.run(main)
