from RBM import RBM


class DBN():
    def __init__(self, hidden_layer_structure, optimization_algorithm='sgd', learning_rate=0.3, num_epochs=10):
        self.RBM_layers = [RBM(num_hidden_units=num_hidden_units, optimization_algorithm=optimization_algorithm,
                               learning_rate=learning_rate, num_epochs=num_epochs) for num_hidden_units in
                           hidden_layer_structure]

    def fit(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)
        return self

    def transform(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
        return input_data