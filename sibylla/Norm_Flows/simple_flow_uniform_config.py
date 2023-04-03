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

"""Config file for a simple normalising flow that uses alternating binary mask coupling layers."""

from typing import Sequence
import jax.numpy as jnp
from ml_collections import config_dict
import haiku as hk
import numpy as np
import distrax

Array = jnp.ndarray


def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    return hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)),
        hk.nets.MLP(hidden_sizes, activate_final=True),
        # We initialize this linear layer to zero so that the flow is initialized
        # to the identity function.
        hk.Linear(
            np.prod(event_shape) * num_bijector_params,
            w_init=jnp.zeros,
            b_init=jnp.zeros),
        hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
    ])


def make_flow_model(event_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int],
                    num_bins: int) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(
            params, range_min=0., range_max=1.)

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes,
                                         num_bijector_params))
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        distrax.Uniform(
            low=jnp.zeros(event_shape),
            high=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape))

    return distrax.Transformed(base_distribution, flow)


def get_config(dataset_name : str) -> config_dict.ConfigDict:
    """Returns the config.

    The config stores information about the:
     - model: how to make it, what constructors to make it from etc.
     - dataset: what dataset to use
     - training: hyperparameters

    """

    n_bins = 4

    if dataset_name == "MNIST":
        data_shape = (28, 28, 1)
    else:
        raise NotImplementedError("dataset not found")

    config = config_dict.ConfigDict()
    config.model_name = "simple_flow_uniform" + dataset_name
    config.data_shape = data_shape
    config.model = dict(
        constructor=make_flow_model,
        kwargs=dict(
            event_shape=data_shape,
            num_layers=12,
            hidden_sizes=[500] * 3,
            num_bins=n_bins
        )
    )
    config.train = dict(
        batch_size=128,
        learning_rate=1e-4,
        # learning_rate_decay_steps=[250000, 500000],
        # learning_rate_decay_factor=0.1,
        seed=42,
        max_gradient_norm=10000.,
    )
    config.eval = dict(
        eval_every=10,
        batch_size=128,
        save_on_eval=True,
    )
    return config
