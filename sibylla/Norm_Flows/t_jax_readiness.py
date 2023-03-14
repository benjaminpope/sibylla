"""tests to see when GPU breaks 

known issue: tensorflow reads wrong cuda version
    - solution: move to torch data loader?
"""

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
# import tensorflow_datasets as tfds
from torch.utils import data
from torchvision.datasets import MNIST
# from ModelStorage import ModelStorage
print(jax.device_count(), jax.devices())

print(jnp.zeros([5]))
jax.random.PRNGKey(3)
exit()
# import all configs
import simple_flow_config
import simple_flow_config_v2

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('flow_model', 'simple_flow',
                  ['simple_flow', 'simple_flow_v2'], 'Flow to train')
flags.DEFINE_enum('dataset', 'MNIST',
                  ['MNIST'], 'Dataset to train')
flags.DEFINE_integer('num_iterations', int(250), 'Number of training steps.')

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
    if FLAGS.flow_model == "simple_flow":
        config = simple_flow_config.get_config(FLAGS.dataset)
    elif FLAGS.flow_model == "simple_flow_v2":
        config = simple_flow_config_v2.get_config(FLAGS.dataset)
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
        data = prepare_data(batch, prng_key)
        # Loss is average negative log likelihood.
        loss = -jnp.mean(log_prob.apply(params, data))
        return loss

    train_ds = load_dataset(tfds.Split.TRAIN, config.train.batch_size)
    eval_ds = load_dataset(tfds.Split.TEST, config.eval.batch_size)

    print(f'Initialising system {FLAGS.flow_model} on dataset {FLAGS.dataset}')
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
    ModelStorage.save_config(save_path, config)

    for step in range(FLAGS.num_iterations):
        dset_imgs = next(train_ds)
        params, opt_state = update(params, rng_key, opt_state, dset_imgs)

        if step % config.eval.eval_every == 0:
            val_loss = eval_fn(params, next(eval_ds))
            train_loss = loss_fn(params, rng_key, dset_imgs)
            logging.info("STEP: %5d; Train loss: %.3f; Validation loss: %.3f", step, train_loss, val_loss)

            if config.eval.save_on_eval:
                ModelStorage.save_checkpoint(save_path, step, params)

    print('Saving model')
    ModelStorage.save_model(save_path, config, params)
    print('Done')


if __name__ == '__main__':
    app.run(main)
