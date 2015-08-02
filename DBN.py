import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from RBM import RBM


class DBN(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, hidden_layers_structure=[50, 50, 200], optimization_algorithm='sgd', learning_rate=0.1,
                 max_iter_backprop=100, lambda_param=0.0, max_epochs_rbm=10, contrastive_divergence_iter=1,
                 cost_func='cross_entropy'):
        self.hidden_layers_structure = hidden_layers_structure
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.max_iter_backprop = max_iter_backprop
        self.lambda_param = lambda_param
        self.max_epochs_rbm = max_epochs_rbm
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.cost_func = cost_func

    def fit(self, data, labels=None):
        # Initialize rbm layers
        self.RBM_layers = list()
        for num_hidden_units in self.hidden_layers_structure:
            rbm = RBM(num_hidden_units=num_hidden_units, optimization_algorithm=self.optimization_algorithm,
                      learning_rate=self.learning_rate, max_epochs=self.max_epochs_rbm,
                      contrastive_divergence_iter=self.contrastive_divergence_iter)
            self.RBM_layers.append(rbm)

        # Fit RBM
        input_data = data
        for rbm in self.RBM_layers:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)

        # Fine-tuning with labels
        if labels is not None:
            self.__fine_tuning(data, labels)
        return self

    def transform(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
        return input_data

    def predict(self, data):
        transformed_data = self.transform(data)
        if len(data.shape) is 1:  # It is a single sample
            sample = transformed_data
            return self.__compute_output_units(sample)
        predicted_data = self.__compute_output_units_matrix(transformed_data)
        labels = np.argmax(predicted_data, axis=1)
        return labels

    def __compute_activations(self, sample):
        input_data = sample
        layers_activation = list()

        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
            layers_activation.append(input_data)

        # Computing activation of output layer
        input_data = self.__compute_output_units(input_data)
        layers_activation.append(input_data)

        return layers_activation

    def __stochastic_gradient_descent(self, _data, _labels):
        data = np.copy(_data)
        labels = np.copy(_labels)
        matrix_error = np.zeros([len(_data), self.num_classes])
        num_samples = len(_data)

        if self.cost_func is 'cross_entropy':
            compute_delta_method = self.__compute_delta_cross_entropy_cost
        else:
            compute_delta_method = self.__compute_delta_quadratic_cost

        for iteration in range(1, self.max_iter_backprop + 1):
            idx = np.random.permutation(len(data))
            data = data[idx]
            labels = labels[idx]
            i = 0
            for sample, label in zip(data, labels):
                delta_W, delta_bias, error_vector = self.__backpropagation(sample, label, compute_delta_method)
                # Updating parameters of hidden layers
                layer = 0
                for rbm in self.RBM_layers:
                    rbm.W = (1 - (self.learning_rate * self.lambda_param) / num_samples) * rbm.W - self.learning_rate * \
                                                                                                   delta_W[layer]
                    rbm.c -= self.learning_rate * delta_bias[layer]
                    layer += 1
                # Updating parameters of output layer
                self.W = (1 - (self.learning_rate * self.lambda_param) / num_samples) * self.W - self.learning_rate * \
                                                                                                 delta_W[layer]
                self.b -= self.learning_rate * delta_bias[layer]
                matrix_error[i, :] = error_vector
                i += 1
            error = np.sum(np.linalg.norm(matrix_error, axis=1))
            print ">> Epoch %d finished \tPrediction error %f" % (iteration, error)

    def __backpropagation(self, input_vector, label, compute_delta):
        x, y = input_vector, label
        deltas = list()
        list_layer_weights = list()
        for rbm in self.RBM_layers:
            list_layer_weights.append(rbm.W)
        list_layer_weights.append(self.W)

        # Forward pass
        layers_activation = self.__compute_activations(input_vector)

        # Backward pass: computing deltas
        activation_output_layer = layers_activation[-1]
        delta_output_layer = compute_delta(y, activation_output_layer)
        deltas.append(delta_output_layer)
        layer_idx = range(len(self.RBM_layers))
        layer_idx.reverse()
        delta_previous_layer = delta_output_layer
        for layer in layer_idx:
            neuron_activations = layers_activation[layer]
            W = list_layer_weights[layer + 1]
            delta = np.dot(delta_previous_layer, W) * (neuron_activations * (1 - neuron_activations))
            deltas.append(delta)
            delta_previous_layer = delta
        deltas.reverse()

        # Computing gradients
        layers_activation.pop()
        layers_activation.insert(0, input_vector)
        layer_gradient_weights = list()
        layer_gradient_bias = list()
        for layer in range(len(list_layer_weights)):
            neuron_activations = layers_activation[layer]
            delta = deltas[layer]
            gradient_W = np.outer(delta, neuron_activations)
            layer_gradient_weights.append(gradient_W)
            layer_gradient_bias.append(delta)

        return layer_gradient_weights, layer_gradient_bias, np.abs(y - activation_output_layer)

    def __fine_tuning(self, data, _labels):
        self.num_classes = len(np.unique(_labels))
        self.W = 0.01 * np.random.randn(self.num_classes, self.RBM_layers[-1].num_hidden_units)
        self.b = 0.01 * np.random.randn(self.num_classes)

        labels = self.__transform_labels_to_network_format(_labels)

        if self.optimization_algorithm is 'sgd':
            self.__stochastic_gradient_descent(data, labels)

    def __transform_labels_to_network_format(self, labels):
        new_labels = np.zeros([len(labels), self.num_classes])
        i = 0
        for label in labels:
            new_labels[i][label] = 1
            i += 1
        return new_labels

    def __compute_output_units(self, vector_visible_units):
        v = vector_visible_units
        return RBM.sigmoid(np.dot(self.W, v) + self.b)

    def __compute_output_units_matrix(self, matrix_visible_units):
        return np.transpose(RBM.sigmoid(np.dot(self.W, np.transpose(matrix_visible_units)) + self.b[:, np.newaxis]))

    def __compute_delta_cross_entropy_cost(self, label, predicted):
        return -(label - predicted)

    def __compute_delta_quadratic_cost(self, label, predicted):
        return -(label - predicted) * (predicted * (1 - predicted))