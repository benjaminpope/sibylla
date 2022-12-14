"""
Abstract base class to work with data loading
"""

import torch
import jax.numpy as np
from torchvision.datasets import MNIST


class DataLoader:
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [DataLoader.numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.int32)
        img = img[..., None]  # Make image [28, 28, 1]
        return img

    def load_MNIST():
        dset_path = './data'
        MNIST_kwargs = {'root' : dset_path,
                        'transform' : DataLoader.image_to_numpy,
                        'download' : True}

        train_dataset = MNIST(train=True, **MNIST_kwargs)

        train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                           generator=torch.Generator().manual_seed(42))

        test_set = MNIST(train=False, **MNIST_kwargs)

        return train_set, val_set, test_set

    def generate_data_loaders(train_set, val_set, test_set):
        train_exmp_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=256,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        collate_fn=DataLoader.numpy_collate)
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=128,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        collate_fn=DataLoader.numpy_collate,
                                                        num_workers=8,
                                                        persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=64,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=4,
                                                 collate_fn=DataLoader.numpy_collate)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=4,
                                                  collate_fn=DataLoader.numpy_collate)
        return train_exmp_loader, train_data_loader, val_loader, test_loader
