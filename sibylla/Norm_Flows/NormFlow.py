import haiku as hk
import jax.numpy as np
from typing import Mapping
from ImageDataset import ImageDataset

Array = np.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]

import sibylla.Norm_Flows.uniform_base_flow_config as uniform_base_flow_config

class NormFlow:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        return self.config.model['constructor'](
            **self.config.model['kwargs'])

    @hk.without_apply_rng
    @hk.transform
    def log_prob(self, data: Array) -> Array:
        model = self.create_model()
        return model.log_prob(data)

    def loss_fn(self, params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
        data = ImageDataset.normalize_dequant_data(batch, prng_key)
        # Loss is average negative log likelihood.
        loss = -np.mean(self.log_prob.apply(params, data))
        return loss
        
if __name__ == "__main__":
    config = uniform_base_flow_config.get_config('MNIST')

    nf = NormFlow(config)

    model = nf.create_model()
