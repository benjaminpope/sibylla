"""
A functional way to share common code by having a function return common functions 
for using norm flows
"""
import haiku as hk
import jax.numpy as np
from typing import Mapping
import jax
from sibylla.Norm_Flows.ImageDataset import ImageDataset

Array = np.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]

import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config

class NormFlow:
    def get_create_model(config):
        def create_model():
            return config.model['constructor'](
                **config.model['kwargs'])
    
        return create_model


    def get_log_prob(config):
        create_model = NormFlow.get_create_model(config)
        
        @hk.without_apply_rng
        @hk.transform
        def log_prob(data: Array) -> Array:
            model = create_model()
            return model.log_prob(data)

        return log_prob
    
    def get_loss_fn(config):
        log_prob = NormFlow.get_log_prob(config)

        def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
            data = ImageDataset.normalize_dequant_data(batch, prng_key)
            # Loss is average negative log likelihood.
            loss = -np.mean(log_prob.apply(params, data))
            return loss

        return loss_fn
    
    def get_inverse_model(config):
        create_model = NormFlow.get_create_model(config)

        @hk.without_apply_rng
        @hk.transform
        def inverse_model(data):
            model = create_model()
            return model.bijector.inverse(data)

        return inverse_model


    def get_forward_model(config):
        create_model = NormFlow.get_create_model(config)

        @hk.without_apply_rng
        @hk.transform
        def forward_model(data):
            model = create_model()
            return model.bijector.forward(data)

        return forward_model

    def get_base_distribution(config):
        create_model = NormFlow.get_create_model(config)

        @hk.without_apply_rng
        @hk.transform
        def _base_distribution():
            model = create_model()
            return model.distribution

        return _base_distribution

    def get_eval_fn(config):
        log_prob = NormFlow.get_log_prob(config)

        @jax.jit
        def eval_fn(params: hk.Params, batch: Batch) -> Array:
            data = ImageDataset.normalize_dequant_data(batch)
            loss = -np.mean(log_prob.apply(params, data))
            return loss
        
        return eval_fn
    
    

if __name__ == "__main__":
    import jax
    config = uniform_base_flow_config.get_config('MNIST')

    log_prob = NormFlow.get_log_prob(config)

    params = log_prob.init(jax.random.PRNGKey(2), np.zeros((1, *config.data_shape)))
    print(log_prob.apply(params, jax.numpy.zeros((5, *config.data_shape))))