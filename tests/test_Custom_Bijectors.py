import jax.numpy as np
import jax
import pytest


from sibylla.Norm_Flows import Squeeze, IgnorantMaskedCoupling


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
    
    def test_inverse_simple(self):
        true = np.arange(1.,17).reshape(1, 4, 4, 1)

        y = np.array([[[[ 1, 3],
                           [ 9,11]],
                          [[ 2, 4],
                           [10,12]],
                          [[ 5, 7],
                           [13,15]],
                          [[ 6, 8],
                           [14,16]]]]).transpose(0,2,3,1)
        
        x,ldet = Squeeze().inverse_and_log_det(y)

        assert true.shape == (1,4,4,1)

        assert ldet == 0
        assert x.shape == (1,4,4,1)

        assert (x == true).all()
        
    
    def test_inverse_block(self):
        n_repeats = 5
        true = np.arange(1.,17).reshape(1, 4, 4, 1)
        true = true.repeat(n_repeats, 0)

        y = np.array([[[[ 1, 3],
                           [ 9,11]],
                          [[ 2, 4],
                           [10,12]],
                          [[ 5, 7],
                           [13,15]],
                          [[ 6, 8],
                           [14,16]]]]).transpose(0,2,3,1)
        y = y.repeat(n_repeats, 0)

        x,ldet = Squeeze().inverse_and_log_det(y)

        assert true.shape == (n_repeats,4,4,1)

        assert ldet == 0
        assert x.shape == (n_repeats,4,4,1)

        assert (x == true).all()

    def test_multiple_applications(self):
        # try with a random matrix and make sure forward and inverse get back to the same
        prng = jax.random.PRNGKey(0)
        x = jax.random.randint(prng, (4,10,8,3), 0, 256)

        bij = Squeeze()

        y = bij.forward(x)
        x_new = bij.inverse(y)

        assert np.isclose(x_new, x).all()
        assert x.shape == x_new.shape
    


class TestIgnorantMaskedCoupling:
    def test_ctor(self):
        event_shape = (1,4,4,1)

        coupling_mask = np.arange(0, np.prod(event_shape)) % 3 == 0
        coupling_mask = np.reshape(coupling_mask, event_shape)


        ignorance_mask = np.arange(0, np.prod(event_shape)) % 3 == 1
        ignorance_mask = np.reshape(ignorance_mask, event_shape)

        IgnorantMaskedCoupling(coupling_mask, ignorance_mask)