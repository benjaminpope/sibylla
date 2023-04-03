import jax.numpy as np
import pytest


from sibylla.Norm_Flows import LearningCurve


class TestLearningCurve():
    def test_load_and_save(self):
        l = LearningCurve([1,2],np.array([[1,2],[1,2]]),["t","e"])
        l.save_model_learning('trained_models/')
        L = LearningCurve.load_model_learning('trained_models/')
        assert((L.epochs == np.array([1,2])).all())
        assert((L.losses == np.array([[1,2],[1,2]])).all())
        assert(L.labels == ["t","e"])
