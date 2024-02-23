"""
Class for managing interfaces with different datasets. Currently supports:
MNIST
"""

from typing import Any, Mapping, Optional, Iterator
import abc
import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

__all__ = ["ImageDataset", "tfdsDataset"]

class ImageDataset(abc.ABC):
    @abc.abstractclassmethod
    def load() -> Iterator[Batch]:
        pass

    def normalize_dequant_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
        """
            Normalize and dequantize image data by adding uniform noise if prng_key is not none and converting [0,256) -> [0,1) 

            batch: A batchof images
        """
        data = batch["image"].astype(np.float32)
        if prng_key is not None:
            # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
            data += jax.random.uniform(prng_key, data.shape)
        return data / 256.  # Normalize pixel values from [0, 256) to [0, 1).

    def get_train_test_iterators(dset_name: str, train_batch_size: int, test_batch_size: int) -> Iterator[Batch]:
        """
            get the iterators for a dataset
            dset_name: e.g. 'mnist', see tfds for full list of currently supported
            train_batch_size: number of images from the train set to use in a batch
            test_batch_size: number of images from the test set to use in a batch
        """
        ds_train = tfdsDataset.get_generator(dset_name.lower(), tfds.Split.TRAIN, train_batch_size)
        ds_test  = tfdsDataset.get_generator(dset_name.lower(), tfds.Split.TEST, test_batch_size)
        return ds_train, ds_test

class tfdsDataset(ImageDataset):
    def get_ds(name: str, split: tfds.Split) -> tf.data.Dataset:
        return tfds.load(name, split=split, shuffle_files=True, with_info=True)

    def get_generator(name: str, split: tfds.Split, batch_size: int) -> Iterator[Batch]:
        ds, _ = tfdsDataset.get_ds(name, split)
        ds = ds.shuffle(buffer_size=10 * batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=5)
        ds = ds.repeat()
        return iter(tfds.as_numpy(ds))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds, ds_info = tfdsDataset.get_ds('emnist', tfds.Split.TRAIN)
    print(type(ds))
    fig = tfds.show_examples(ds, ds_info)
    plt.show()