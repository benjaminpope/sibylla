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

import matplotlib.pyplot as plt
import simple_flow_config

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


def main(_):
    system = FLAGS.system
    if True:
        config = simple_flow_config.get_config('MNIST')
    else:
        raise KeyError(system)

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

    def display_fwd_inv(params, img):
        fwd = forward_model.apply(params, img)
        inv = inverse_model.apply(params, img)

        print(f"Norms of: img {jnp.linalg.norm(img)}, fwd {jnp.linalg.norm(fwd)}, inv {jnp.linalg.norm(inv)}")
        plt.subplot(131)
        plt.imshow(img, vmin=0, vmax=1)
        plt.subplot(132)
        plt.imshow(fwd, vmin=0, vmax=1)
        plt.subplot(133)
        plt.imshow(inv, vmin=0, vmax=1)
        plt.show()
    
    def generate_distance_hist(ds, params, prng_seq):
        imgs = prepare_data(next(ds), next(prng_seq))
        fwd = forward_model.apply(params, imgs)
        distances = np.zeros((imgs.shape[0],))
        # TODO: optimize
        for idx in range(fwd.shape[0]):
            distances[idx] = jnp.linalg.norm(fwd[idx,:,:,:])
        plt.hist(distances, label="eval")
        
        # inverse
        imgs = 1 - imgs
        fwd = forward_model.apply(params, imgs)
        distances = np.zeros((imgs.shape[0],))
        # TODO: optimize
        for idx in range(fwd.shape[0]):
            distances[idx] = jnp.linalg.norm(fwd[idx,:,:,:])
        plt.hist(distances, label="inverse")
        
        # noise
        noise = jax.random.normal(next(prng_seq), imgs.shape)
        fwd = forward_model.apply(params, noise)
        distances = np.zeros((imgs.shape[0],))
        # TODO: optimize
        for idx in range(fwd.shape[0]):
            distances[idx] = jnp.linalg.norm(fwd[idx,:,:,:])
        print(distances)
        plt.hist(distances, label="noise")
        
        plt.legend()
        plt.show()
        
    
    eval_ds = load_dataset(tfds.Split.TEST, config.eval.batch_size)

    # load params
    params = ModelStorage.load_model(save_path)

    prng_seq = hk.PRNGSequence(42)


    # generate_distance_hist(eval_ds, params, prng_seq)

    img = prepare_data(next(eval_ds), next(prng_seq))[0]
    display_fwd_inv(params, img)
    noise = jax.random.normal(next(prng_seq), img.shape)
    display_fwd_inv(params, noise)


if __name__ == '__main__':
    app.run(main)
