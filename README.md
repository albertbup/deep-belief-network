# DeepBeliefNet
A simple Python implementation of Deep Belief Network based on 
> Fischer, Asja, and Christian Igel. "Training restricted Boltzmann machines: an introduction." Pattern Recognition 47.1 (2014): 25-39.

## Usage
This implementation follows the scikit-learn way to work:
  
    # Create a DBN with three layers containing 200, 200 and 500 hidden units respectively
    dbn = DBN([200, 200, 500])
    
    # Learn from data in X
    dbn.fit(X, optimization_algorithm='sgd', learning_rate=0.1, num_epochs=10)
    
    # Transform data in X
    X_transformed = dbn.transform(X)
