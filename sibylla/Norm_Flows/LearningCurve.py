import os
import jax.numpy as np
import pickle
import matplotlib.pyplot as plt

class LearningCurve:
    """
        data that takes care of the train and test loss tracking
    """

    def __init__(self, epochs, train_losses, test_losses) -> None:
        assert len(epochs) == len(train_losses)
        assert len(epochs) == len(test_losses)

        self.epochs = np.array(epochs)
        self.train_losses = np.array(train_losses)
        self.test_losses = np.array(test_losses)

    def save_model_learning(self, save_path):
        """
            save the learning curve data for the model
        """
        with open(os.path.join(save_path, 'learning_data.pickle'), 'wb') as f:
            pickle.dump(self, f)
        

    def plot_model_learning(self):
        plt.figure()
        plt.plot(self.epochs, self.train_losses, label='training')
        plt.plot(self.epochs, self.test_losses, label='testing')
        plt.show()


    def load_model_learning(save_path):
        with open(os.path.join(save_path, 'learning_data.pickle'), 'rb') as f:
            learning_data = pickle.load(f)
        return learning_data

