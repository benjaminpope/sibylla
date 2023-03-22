import os
import jax.numpy as np
import pickle
import matplotlib.pyplot as plt

__all__ = ["LearningCurve"]


class LearningCurve:
    """
        data that takes care of the train and test loss tracking
    """

    def __init__(self, epochs, losses, labels) -> None:
        assert len(epochs) == losses.shape[1]
        assert len(labels) == losses.shape[0]

        self.epochs = np.array(epochs)
        self.losses = np.array(losses)
        self.labels = labels

    def save_model_learning(self, save_path):
        """
            save the learning curve data for the model
        """
        with open(os.path.join(save_path, 'learning_data.pickle'), 'wb') as f:
            pickle.dump(self, f)
        

    def plot_model_learning(self, save_path=None):
        plt.figure()
        for label_idx, label in enumerate(self.labels):
            plt.plot(self.epochs, self.losses[label_idx,:], label=label)
        plt.legend()
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        
        if save_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path,'learning_curve.pdf'))


    def load_model_learning(save_path):
        with open(os.path.join(save_path, 'learning_data.pickle'), 'rb') as f:
            learning_data = pickle.load(f)
        return learning_data

