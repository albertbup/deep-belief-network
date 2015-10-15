# deep-belief-network
A simple, clean Python implementation of Deep Belief Networks with sigmoid units based on binary Restricted Boltzmann Machines (RBM):
> Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.

> Fischer, Asja, and Christian Igel. "Training restricted Boltzmann machines: an introduction." Pattern Recognition 47.1 (2014): 25-39.

## Usage
This implementation follows [scikit-learn](http://scikit-learn.org) guidelines and in turn, can be used alongside it. Next you have a demo code for solving digits classification problem which can be found in **classification_demo.py** (check **regression_demo.py** for a regression problem example).
    
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics.classification import accuracy_score
    import numpy as np
    
    from dbn.models import SupervisedDBNClassification
    
    
    # Loading dataset
    digits = load_digits()
    X, Y = digits.data, digits.target
    X = X.astype(np.float32)
    
    # 0-1 scaling
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    
    # Splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256], learning_rate_rbm=0.01,
                                             learning_rate=0.001, max_epochs_rbm=30, max_iter_backprop=1000,
                                             lambda_param=0.0)
    classifier.fit(X_train, Y_train)
    
    # Test
    Y_pred = classifier.predict(X_test)
    print 'Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)
    
## Requierements
NumPy and scikit-learn packages must be installed in the system. If aren't, simply do:
    
    pip install numpy
    pip install scikit-learn
  
## Installation

    pip install git+git://github.com/albertbup/deep-belief-network.git@master
