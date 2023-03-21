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

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


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




class MNIST(ImageDataset):
    def load(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
        ds = tfds.load("mnist", split=split, shuffle_files=True)
        ds = ds.shuffle(buffer_size=10 * batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=5)
        ds = ds.repeat()
        return iter(tfds.as_numpy(ds))

if __name__ == "__main__":
    print(type(MNIST.load(tfds.Split.TRAIN,12)))

