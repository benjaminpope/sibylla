"""Evaluates a trained flow on data"""


from typing import Callable, Dict, Tuple, Union
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import os
import pickle
import json
from ModelStorage import ModelStorage

import simple_flow_config 

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('system', 'simple_MNIST',
                  ['simple_MNIST'], 'Experiment and dataset to train')
flags.DEFINE_integer('num_iterations', int(20), 'Number of training steps.')

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
    ds = tfds.load("mnist", split=split, shuffle_files=True)
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

    save_path = ModelStorage.get_model_path(config)
    
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
    
    eval_ds = load_dataset(tfds.Split.TEST, config.eval.batch_size)

    # load params
    with open(os.path.join(save_path, 'model.pickle'),'rb') as f:
        params = pickle.load(f)

    prng_seq = hk.PRNGSequence(42)
    
    import matplotlib.pyplot as plt
    imgs = prepare_data(next(eval_ds), next(prng_seq))
    img = imgs[0]
    
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
    
    noise = jax.random.normal(next(prng_seq), img.shape)
    fwd = forward_model.apply(params, noise)    
    inv = inverse_model.apply(params, noise)
    
    print(f"Norms of: noise {jnp.linalg.norm(noise)}, fwd {jnp.linalg.norm(fwd)}, inv {jnp.linalg.norm(inv)}")
    plt.subplot(131)
    plt.imshow(noise, vmin=0, vmax=1)
    plt.subplot(132)
    plt.imshow(fwd, vmin=0, vmax=1)
    plt.subplot(133)
    plt.imshow(inv, vmin=0, vmax=1)
    plt.show()

if __name__ == '__main__':
    app.run(main)
