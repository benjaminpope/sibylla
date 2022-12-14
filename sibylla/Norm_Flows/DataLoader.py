"""
Abstract base class to work with data loading
"""

import numpy


class DataLoader:
    def numpy_collate(batch):
        if isinstance(batch[0], numpy.ndarray):
            return numpy.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [DataLoader.numpy_collate(samples) for samples in transposed]
        else:
            return numpy.array(batch)
