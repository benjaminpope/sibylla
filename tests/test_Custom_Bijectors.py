import jax.numpy as np
import jax
import pytest
import distrax
from typing import Sequence
import haiku as hk

from sibylla.Norm_Flows import Squeeze, IgnorantMaskedCoupling


class TestSqueeze():
    def test_constructor(self):
        Squeeze(4,4)
    
    def test_forward_simple(self):
        bijec = Squeeze(4,4)

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
        bijec = Squeeze(4,4)

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
        
        x,ldet = Squeeze(4,4).inverse_and_log_det(y)

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

        x,ldet = Squeeze(4,4).inverse_and_log_det(y)

        assert true.shape == (n_repeats,4,4,1)

        assert ldet == 0
        assert x.shape == (n_repeats,4,4,1)

        assert (x == true).all()

    def test_multiple_applications(self):
        # try with a random matrix and make sure forward and inverse get back to the same
        prng = jax.random.PRNGKey(0)
        x = jax.random.randint(prng, (4,10,8,3), 0, 256)

        bij = Squeeze(4,4)

        y = bij.forward(x)
        x_new = bij.inverse(y)

        assert np.isclose(x_new, x).all()
        assert x.shape == x_new.shape
    


class NoLongerTestIgnorantMaskedCoupling:
    def _get_typical_inputs(self, event_shape):
        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=0., range_max=1.)
        
        @hk.without_apply_rng
        @hk.transform
        def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Sequential:
            """Creates an MLP conditioner for each layer of the flow."""
            return hk.Sequential([
                hk.Flatten(preserve_dims=-len(event_shape)),
                hk.nets.MLP(hidden_sizes, activate_final=True),
                # We initialize this linear layer to zero so that the flow is initialized
                # to the identity function.
                hk.Linear(
                    np.prod(event_shape) * num_bijector_params,
                    w_init=np.zeros,
                    b_init=np.zeros),
                hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
            ])

        params = make_conditioner.init(jax.random.PRNGKey(1), event_shape, [5]*3, 4)

        return bijector_fn, make_conditioner.apply(params, event_shape, [5]*3, 4)

    def test_ctor(self):
        event_shape = np.array((1,4,4,1))

        coupling_mask = np.arange(0, np.prod(event_shape)) % 3 == 0
        coupling_mask = np.reshape(coupling_mask, event_shape)

        ignorance_mask = np.arange(0, np.prod(event_shape)) % 3 == 1
        ignorance_mask = np.reshape(ignorance_mask, event_shape)

        bij, cond = self._get_typical_inputs(event_shape)

        IgnorantMaskedCoupling(coupling_mask, ignorance_mask, cond, bij)

    def test_ctor_reject_overlap(self):
        # verify that it is invalid to use two masks with any overlap
        event_shape = np.array((1,4,4,1))
        coupling_mask = np.arange(0, np.prod(event_shape)) % 3 == 0
        coupling_mask = np.reshape(coupling_mask, event_shape)

        ignorance_mask = coupling_mask.copy()
        
        with pytest.raises(ValueError):
            IgnorantMaskedCoupling(coupling_mask, ignorance_mask)

    def test_forward(self):
        # rather than testing for a simple bijector exactly, we test for the emergent behaviour 
        # with a complex setup i.e. checking that changing the values that are coupling masked
        # does change the output while changing the values that are ignorance masked has no effect
        event_shape = np.array((1,4,4,1))

        coupling_mask = np.arange(0, np.prod(event_shape)) % 3 == 0
        coupling_mask = np.reshape(coupling_mask, event_shape)


        ignorance_mask = np.arange(0, np.prod(event_shape)) % 3 == 1
        ignorance_mask = np.reshape(ignorance_mask, event_shape)


        imc = IgnorantMaskedCoupling(coupling_mask, ignorance_mask)

        
