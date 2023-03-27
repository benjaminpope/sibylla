"""
A functional way to share common code by having a function return common functions 
for using norm flows
"""
import haiku as hk
import jax.numpy as np
from typing import Mapping
from ImageDataset import ImageDataset

Array = np.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]

import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config

class NormFlow:
    def get_log_prob(config):
        def create_model():
            return config.model['constructor'](
                **config.model['kwargs'])

        @hk.without_apply_rng
        @hk.transform
        def log_prob(data: Array) -> Array:
            model = create_model()
            return model.log_prob(data)

        return log_prob
        
if __name__ == "__main__":
    import jax
    config = uniform_base_flow_config.get_config('MNIST')

    log_prob = NormFlow.get_log_prob(config)

    params = log_prob.init(jax.random.PRNGKey(2), np.zeros((1, *config.data_shape)))
    print(log_prob.apply(params, jax.numpy.zeros((5, *config.data_shape))))