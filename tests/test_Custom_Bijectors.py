import jax.numpy as np
import pytest


from sibylla.Norm_Flows import Squeeze


class TestSqueeze():
    def test_constructor(self):
        Squeeze()
    
    def test_forward_simple(self):
        bijec = Squeeze()

        x = np.arange(1.,17).reshape(1, 4, 4, 1)

        y, ldet = bijec.forward_and_log_det(x)

        # the actual answer
        true = np.array([[[[ 1, 3],
                           [ 9,11]],
                          [[ 2, 4],
                           [10,12]],
                          [[ 5, 7],
                           [13,15]],
                          [[ 6, 8],
                           [14,16]]]]).transpose(0,2,3,1)
        assert true.shape == (1,2,2,4)

        assert ldet == 0
        assert y.shape == (1,2,2,4)

        assert (y == true).all()

    def test_forward_block(self):
        bijec = Squeeze()

        # same as test_forward_simple but with a block size of 5
        n_repeats = 5
        x = np.arange(1.,17).reshape(1, 4, 4, 1)
        x = x.repeat(n_repeats, 0)

        # the actual answer
        true = np.array([[[[ 1, 3],
                           [ 9,11]],
                          [[ 2, 4],
                           [10,12]],
                          [[ 5, 7],
                           [13,15]],
                          [[ 6, 8],
                           [14,16]]]]).transpose(0,2,3,1)
        true = true.repeat(n_repeats,0)

        y, ldet = bijec.forward_and_log_det(x)
        assert true.shape == (n_repeats,2,2,4)

        assert ldet == 0
        assert y.shape == (n_repeats,2,2,4)

        assert (y == true).all()