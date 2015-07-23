import numpy as np

from RBM import RBM


class DBN():
    def __init__(self, hidden_layers_structure, optimization_algorithm='sgd', learning_rate=0.3, num_epochs=10):
        self.RBM_layers = [RBM(num_hidden_units=num_hidden_units, optimization_algorithm=optimization_algorithm,
                               learning_rate=learning_rate, num_epochs=num_epochs) for num_hidden_units in
                           hidden_layers_structure]

    def fit(self, data, labels=None):
        input_data = data
        for rbm in self.RBM_layers:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)

        if labels is not None:
            self.num_classes = len(np.unique(labels))  # TODO Possible bug here
            self.W = np.random.randn(self.num_classes, self.RBM_layers[-1].num_hidden_units)
            self.b = np.random.randn(self.num_classes)
            x = data[0, :]
            y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.__backpropagation(x, y)
            self.__fine_tuning()
        return self

    def transform(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
        return input_data

    def predict(self, data):
        return

    def __compute_activations(self, sample):
        input_data = sample
        layers_activation = list()

        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
            layers_activation.append(input_data)

        # Computing activation of output layer
        input_data = RBM.sigmoid(np.dot(self.W, input_data) + self.b)
        layers_activation.append(input_data)

        return layers_activation

    def __backpropagation(self, input_vector, labels):
        x, y = input_vector, labels
        deltas = list()
        list_layer_weights = list()
        for rbm in self.RBM_layers:
            list_layer_weights.append(rbm.W)
        list_layer_weights.append(self.W)

        # Forward pass
        layers_activation = self.__compute_activations(input_vector)

        # Backward pass: computing deltas
        activation_output_layer = layers_activation[-1]
        delta_output_layer = (activation_output_layer - y)
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
            W = list_layer_weights[layer]
            gradient_W = np.zeros(W.shape)
            neuron_activations = layers_activation[layer]
            for j in range(len(neuron_activations)):
                gradient_W[:, j] = deltas[layer] * neuron_activations[j]  # TODO Vectorize this
            layer_gradient_weights.append(gradient_W)
            layer_gradient_bias.append(deltas[layer])

        return layer_gradient_weights, layer_gradient_bias

    def __fine_tuning(self):
        return