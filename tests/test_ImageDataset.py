import jax.numpy as np
import pytest
import jax.random as random

import types

import tensorflow_datasets as tfds
from sibylla.Norm_Flows import tfdsDataset



class TestImageDataset():
    def test_always(self):
        assert(True)

class Test_tfdsDataset():
    def test_generator(self):
        batch_size = 16
        im_width = 28
        for split in [tfds.Split.TRAIN, tfds.Split.TEST]:
            ds_loader = tfdsDataset.get_generator('mnist', split, batch_size)

            assert isinstance(ds_loader, types.GeneratorType)
            assert (next(ds_loader)['image'].shape == (batch_size, im_width, im_width, 1))
            
            img = next(ds_loader)['image'][0]

            # check that there exists pixels > 1 but less than 256
            assert (np.logical_and(img > 1, img < 256).any())