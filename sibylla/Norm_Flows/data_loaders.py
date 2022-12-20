import torch
import jax.numpy as np
from torchvision.datasets import MNIST

__all__ = ["ImageDataLoader"]


class ImageDataLoader:
    """
    Abstract base class to work with data loading
    """
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [ImageDataLoader.numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.int32)
        img = img[..., None]  # Make image [28, 28, 1]
        return img

    def load_MNIST():
        dset_path = './data'
        MNIST_kwargs = {'root' : dset_path,
                        'transform' : ImageDataLoader.image_to_numpy,
                        'download' : True}

        train_dataset = MNIST(train=True, **MNIST_kwargs)

        train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                           generator=torch.Generator().manual_seed(42))

        test_set = MNIST(train=False, **MNIST_kwargs)

        return train_set, val_set, test_set

    def generate_data_loaders(train_set, val_set, test_set):
        # TODO: split based off if using pretrained (CPU) or trained and allocate as needed
        Warning("Using simple loader, not to be used for training")
        train_exmp_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=16,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=0,
                                                        collate_fn=ImageDataLoader.numpy_collate)
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=16,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        collate_fn=ImageDataLoader.numpy_collate,
                                                        num_workers=0)
#                                                       # persistent_workers=True
#                                                       # TODO: add this back in for actual training
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=16,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0,
                                                 collate_fn=ImageDataLoader.numpy_collate)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=16,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  collate_fn=ImageDataLoader.numpy_collate)
        return train_exmp_loader, train_data_loader, val_loader, test_loader
