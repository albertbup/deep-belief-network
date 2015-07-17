from RBM import RBM


class DBN():
    def __init__(self, hidden_layer_structure):
        self.hidden_layer_structure = hidden_layer_structure
        self.num_hidden_layers = len(self.hidden_layer_structure)

        self.RBM_layers = [RBM(num_hidden_units) for num_hidden_units in self.hidden_layer_structure]

    def fit(self, data, algorithm='sgd', learning_rate=1.0, epochs=10):
        input_data = data
        for rbm in self.RBM_layers:
            rbm.fit(input_data, algorithm=algorithm, learning_rate=learning_rate, epochs=epochs)
            input_data = rbm.transform(input_data)
        return self

    def transform(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
        return input_data