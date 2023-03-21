import jax.numpy as np
import pytest
import jax.random as random


import sibylla.Norm_Flows.LearningCurve as lc


class TestLearningCurve():
    def test_load_and_save(self):
        l = lc.LearningCurve([1,2],[1,2],[1,2])
        l.save_model_learning('trained_models/')
        L = lc.load_model_learning('trained_models/')
        assert(L.epochs == [1,2])
        assert(L.train_losses == [1,2])
        assert(L.test_losses == [1,2])
