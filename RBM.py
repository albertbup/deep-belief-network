import numpy as np


class RBM():
    def __init__(self, data, num_hidden_units):
        self.num_visible_units = data.shape[0] + 1
        self.num_hidden_units = num_hidden_units + 1
        self.W = np.zeros([self.num_hidden_units, self.num_visible_units])
        self.H = np.zeros(self.num_hidden_units)
        self.V = np.zeros(self.num_visible_units)

    def sigmoid(self, vector):
        func = np.vectorize(self.__sigmoid)
        return func(vector)

    def __sigmoid(self, x):
        return 1 / (1.0 + np.exp(-x))

    def fit(self):
        return

    def transform(self):
        return

    def __contrastive_divergence(self, k=1):
        return

    def __compute_hidden_units(self):
        return

    def __compute_visible_units(self):
        return