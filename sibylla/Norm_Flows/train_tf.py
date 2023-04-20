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

from typing import Tuple, Union, Any, Iterator, Mapping, Optional

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
from ImageDataset import ImageDataset
from NormFlow import NormFlow

# import all configs
import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config
import different_mask_flow_uniform_config as different_mask_flow_uniform_config
import simple_flow_uniform_config as simple_flow_uniform_config
import random_mask_flow_uniform_config as random_mask_flow_uniform_config
from LearningCurve import LearningCurve

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('flow_model', 'simple_flow_uniform',
                  ['uniform_base_flow',
                   'different_mask_flow',
                   'random_mask_flow',
                   'simple_flow_uniform'], 'Flow to train')
flags.DEFINE_enum('dataset', 'MNIST',
                  ['MNIST'], 'Dataset to train')
flags.DEFINE_integer('num_iterations', int(2e3), 'Number of training steps.')

FLAGS = flags.FLAGS

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

jax.random.PRNGKey(5)



def main(_):
    if FLAGS.flow_model == "uniform_base_flow":
        config = uniform_base_flow_config.get_config(FLAGS.dataset)
    elif FLAGS.flow_model == "different_mask_flow":
        config = different_mask_flow_uniform_config.get_config(FLAGS.dataset)
    elif FLAGS.flow_model == "random_mask_flow":
        config = random_mask_flow_uniform_config.get_config(FLAGS.dataset)
    elif FLAGS.flow_model == "simple_flow_uniform":
        config = simple_flow_uniform_config.get_config(FLAGS.dataset)
    else:
        raise KeyError(f'{FLAGS.flow_model} is not implemented!')

    save_path = ModelStorage.make_model_path(config)

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
        data = ImageDataset.normalize_dequant_data(batch, prng_key)
        # Loss is average negative log likelihood.
        loss = -jnp.mean(log_prob.apply(params, data))
        return loss


    train_ds, eval_ds = ImageDataset.get_train_test_iterators('mnist', config.train.batch_size, config.eval.batch_size)
    etrain_ds, eeval_ds = ImageDataset.get_train_test_iterators('emnist', config.train.batch_size, config.eval.batch_size)

    logging.info(f"Event size: {next(train_ds)['image'].shape}")

    logging.info(f'Initialising system {FLAGS.flow_model} on dataset {FLAGS.dataset}')
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

    eval_fn = NormFlow.get_eval_fn(config)

    logging.info('Beginning of training...')
    ModelStorage.save_config(save_path, config)


    epochs = []
    eval_dsets = [eval_ds, etrain_ds]
    eval_labels = ["evaluation", "emnist"]
    loss_labels = ["train", *eval_labels]
    losses = np.zeros([len(eval_labels) + 1, len(np.arange(0,FLAGS.num_iterations,config.eval.eval_every))])
    loss_epoch_idx = 0

    for step in range(FLAGS.num_iterations):
        dset_imgs = next(train_ds)
        params, opt_state = update(params, rng_key, opt_state, dset_imgs)

        if step % config.eval.eval_every == 0:
            train_loss = loss_fn(params, rng_key, dset_imgs)

            epochs.append(step)
            losses[0,loss_epoch_idx] = train_loss
            for i, eval_ds in enumerate(eval_dsets):
                val_loss = eval_fn(params, next(eval_ds))
                logging.info("STEP: %5d; Train loss: %.3f; Validation loss (%s): %.3f", step, train_loss, eval_labels[i], val_loss)
                losses[i+1,loss_epoch_idx] = val_loss
            loss_epoch_idx += 1
            

            if config.eval.save_on_eval:
                ModelStorage.save_checkpoint(save_path, step, params)

    logging.info('Saving model')
    ModelStorage.save_model(save_path, config, params)
    curve = LearningCurve(epochs, losses, loss_labels)
    curve.save_model_learning(save_path)
    curve.plot_model_learning(save_path)
    logging.info('Done')
    
if __name__ == '__main__':
    app.run(main)
