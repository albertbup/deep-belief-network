import numpy as np


class RBM():
    def __init__(self, num_hidden_units=200, optimization_algorithm='sgd', learning_rate=0.1, max_epochs=10,
                 contrastive_divergence_iter=1):
        self.num_hidden_units = num_hidden_units
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter

    @classmethod
    def sigmoid(cls, vector):
        func = np.vectorize(cls.__sigmoid)
        return func(vector)

    @classmethod
    def __sigmoid(cls, x):
        return 1 / (1.0 + np.exp(-x))

    def fit(self, data):
        # Initialize RBM parameters
        self.num_visible_units = data.shape[1]
        self.W = np.random.randn(self.num_hidden_units, self.num_visible_units)
        self.c = np.random.randn(self.num_hidden_units)
        self.b = np.random.randn(self.num_visible_units)

        if self.optimization_algorithm is 'sgd':
            self.__stochastic_gradient_descent(data)
        return self

    def transform(self, data):
        if len(data.shape) is 1:  # It is a single sample
            sample = data
            return self.__compute_hidden_units(sample)
        transformed_data = np.zeros([data.shape[0], self.num_hidden_units])
        i = 0
        for sample in data:
            transformed_data[i, :] = self.__compute_hidden_units(sample)
            i += 1
        return transformed_data

    def __reconstruct(self, transformed_data):
        reconstructed_data = np.zeros([transformed_data.shape[0], self.num_visible_units])
        i = 0
        for sample in transformed_data:
            reconstructed_data[i, :] = self.__compute_visible_units(sample)
            i += 1
        return reconstructed_data

    def __stochastic_gradient_descent(self, _data):
        data = np.copy(_data)
        for iteration in range(1, self.max_epochs + 1):
            np.random.shuffle(data)
            for sample in data:
                delta_W, delta_b, delta_c = self.__contrastive_divergence(sample)
                self.W += self.learning_rate * delta_W
                self.b += self.learning_rate * delta_b
                self.c += self.learning_rate * delta_c
            error = self.__compute_reconstruction_error(data)
            print ">> Epoch %d finished \tReconstruction error %f" % (iteration, error)

    def __contrastive_divergence(self, vector_visible_units):
        v_0 = vector_visible_units
        v_t = np.copy(v_0)

        # Sampling
        for t in range(self.contrastive_divergence_iter):
            h_t = self.__compute_hidden_units(v_t)
            v_t = self.__compute_visible_units(h_t)

        # Computing deltas
        v_k = v_t
        h_0 = self.__compute_hidden_units(v_0)
        h_k = self.__compute_hidden_units(v_k)
        delta_W = np.outer(h_0, v_0) - np.outer(h_k, v_k)
        delta_b = v_0 - v_k
        delta_c = h_0 - h_k

        return delta_W, delta_b, delta_c

    def __compute_hidden_units(self, vector_visible_units):
        v = vector_visible_units
        return self.__sigmoid(np.dot(self.W, v) + self.c)

    def __compute_visible_units(self, vector_hidden_units):
        h = vector_hidden_units
        return self.__sigmoid(np.dot(h, self.W) + self.b)

    def __compute_free_energy(self, vector_visible_units):
        v = vector_visible_units
        h = self.__compute_hidden_units(v)
        energy = - np.dot(self.b, v) - np.sum(np.log(1 + np.exp(h)))
        return energy

    def __compute_reconstruction_error(self, data):
        data_transformed = self.transform(data)
        data_reconstructed = self.__reconstruct(data_transformed)
        return np.sum(np.linalg.norm(data_reconstructed - data, axis=1))