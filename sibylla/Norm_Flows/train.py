#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a normalising flow """

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

import simple_flow_config 

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('system', 'simple_MNIST',
                  ['simple_MNIST'], 'Experiment and dataset to train')
flags.DEFINE_integer('num_iterations', int(10**2), 'Number of training steps.')

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


def make_model_path(config):
    """ Create a folder to save the model in and return the path"""
    model_path = os.path.join('trained_models', config.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    files = os.listdir(model_path)
    files.sort()
    if files != []:
        last_version = files[-1]
        last_version = last_version[len('version_'):]
        
        this_version = int(last_version) + 1
    else:
        this_version = 0
    
    version_path = os.path.join(model_path, f'version_{this_version}')
    os.makedirs(version_path)
    return version_path


def main(_):
    
    system = FLAGS.system
    if True:
        config = simple_flow_config.get_config('MNIST')
    else:
        raise KeyError(system)

    # lr_schedule_fn = utils.get_lr_schedule(
    #     config.train.learning_rate, config.train.learning_rate_decay_steps,
    #     config.train.learning_rate_decay_factor)
    optimizer = optax.adam(config.train.learning_rate)
    if config.train.max_gradient_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.max_gradient_norm), optimizer)

    def create_model():
        return config.model['constructor'](
            **config.model['kwargs'])

    
    
    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: Array) -> Array:
        model = create_model()
        return model.log_prob(data)
    
    def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
        data = prepare_data(batch, prng_key)
        # Loss is average negative log likelihood.
        loss = -jnp.mean(log_prob.apply(params, data))
        return loss
    
    train_ds = load_dataset(tfds.Split.TRAIN, config.train.batch_size)
    eval_ds = load_dataset(tfds.Split.TEST, config.eval.batch_size)

    print(f'Initialising system {system}')
    rng_key = jax.random.PRNGKey(config.train.seed)

    rng_key, init_key = jax.random.split(rng_key)
    params = log_prob.init(init_key, np.zeros((1, *config.data_shape)))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params: hk.Params,
                         prng_key: PRNGKey,
                         opt_state: OptState,
                         batch: Batch) -> Tuple[hk.Params, OptState]:
        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, prng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
    # jitted_eval = jax.jit(apply_eval_fn)

    @jax.jit
    def eval_fn(params: hk.Params, batch: Batch) -> Array:
        data = prepare_data(batch)  # We don't dequantize during evaluation.
        loss = -jnp.mean(log_prob.apply(params, data))
        return loss

    print('Beginning of training.')
    
    for step in range(FLAGS.num_iterations):
        params, opt_state = update(params, rng_key, opt_state,
                                                             next(train_ds))

        if step % config.eval.eval_every == 0:
            val_loss = eval_fn(params, next(eval_ds))
            logging.info("STEP: %5d; Validation loss: %.3f", step, val_loss)
            
            if config.eval.save_on_eval:
                pass
    print('Done')


if __name__ == '__main__':
    app.run(main)