import numpy as np


class DigitClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        pass # delete this line and replace yours

    def load_model(self):
        pass  # delete this line and replace yours

    def save_model(self):
        pass  # delete this line and replace yours

    def train(self, x, y, **kwargs):
        pass  # delete this line and replace yours

    def predict(self, x_test):
        """
        :param x_test: a numpy array with dimension (N,D)
        :return: a numpy array with dimension (N,)
        """
        return 2 * np.ones(x_test.shape[0])  # delete this line and replace yours
