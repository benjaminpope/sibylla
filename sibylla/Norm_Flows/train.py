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

"""Energy-based training of a flow model on an atomistic system."""

from typing import Callable, Dict, Tuple, Union
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from absl import app
from absl import flags
import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from simple_flow_config import get_config

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('system', 'simple_MNIST',
                  ['simple_MNIST'], 'Experiment and dataset to train')
flags.DEFINE_integer('num_iterations', int(10**1), 'Number of training steps.')

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


def main(_):
    system = FLAGS.system
    if True:
        config = get_config('MNIST')
    else:
        raise KeyError(system)

    # lr_schedule_fn = utils.get_lr_schedule(
    #     config.train.learning_rate, config.train.learning_rate_decay_steps,
    #     config.train.learning_rate_decay_factor)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale(-1))
    if config.train.max_gradient_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.max_gradient_norm), optimizer)

    def create_model():
        return config.model['constructor'](
            **config.model['kwargs'])

    # def loss_fn():
    #     """Loss function for training."""
    #     model = create_model()

    #     loss, stats = _get_loss(
    #         model=model,
    #         energy_fn=energy_fn_train,
    #         beta=state.beta,
    #         num_samples=config.train.batch_size,
    #         )

    #     metrics = {
    #         'loss': loss,
    #         'energy': jnp.mean(stats['energy']),
    #         'model_entropy': -jnp.mean(stats['model_log_prob']),
    #     }
    #     return loss, metrics
    
    
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
    

    print(f'Initialising system {system}')
    rng_key = jax.random.PRNGKey(config.train.seed)
    init_fn, apply_fn = hk.transform(loss_fn)
    _, apply_eval_fn = hk.transform(eval_fn)

    rng_key, init_key = jax.random.split(rng_key)
    params = init_fn(init_key)
    opt_state = optimizer.init(params)

    def _loss(params, rng):
        loss, metrics = apply_fn(params, rng)
        return loss, metrics
    jitted_loss = jax.jit(jax.value_and_grad(_loss, has_aux=True))
    jitted_eval = jax.jit(apply_eval_fn)

    step = 0
    print('Beginning of training.')
    while step < FLAGS.num_iterations:
        # Training update.
        rng_key, loss_key = jax.random.split(rng_key)
        (_, metrics), g = jitted_loss(params, loss_key)
        if (step % 50) == 0:
            print(f'Train[{step}]: {metrics}')
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

        if (step % config.test.test_every) == 0:
            rng_key, val_key = jax.random.split(rng_key)
            metrics = jitted_eval(params, val_key)
            print(f'Valid[{step}]: {metrics}')

        step += 1

    print('Done')


if __name__ == '__main__':
    app.run(main)