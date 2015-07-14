import numpy as np


class RBM():
    def __init__(self, data, num_hidden_units):
        self.num_visible_units = data.shape[0]
        self.num_hidden_units = num_hidden_units
        self.W = np.random.randn(self.num_hidden_units, self.num_visible_units)
        self.H = np.zeros(self.num_hidden_units)
        self.V = np.zeros(self.num_visible_units)
        self.c = np.random.randn(self.num_hidden_units)
        self.b = np.random.randn(self.num_visible_units)

    @classmethod
    def sigmoid(cls, vector):
        func = np.vectorize(cls.__sigmoid)
        return func(vector)

    @classmethod
    def __sigmoid(cls, x):
        return 1 / (1.0 + np.exp(-x))

    def fit(self):
        return

    def transform(self):
        return

    def __contrastive_divergence(self, k=1):
        return

    def __compute_hidden_units(self, matrix_weights, vector_visible_units, vector_bias_hidden):
        W, v, c = matrix_weights, vector_visible_units, vector_bias_hidden
        return self.__sigmoid(np.dot(W, v) + c)

    def __compute_visible_units(self, matrix_weights, vector_hidden_units, vector_bias_visible):
        W, h, b = matrix_weights, vector_hidden_units, vector_bias_visible
        return self.__sigmoid(np.dot(h, W) + b)