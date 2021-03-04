# deep-belief-network
A simple, clean, fast Python implementation of Deep Belief Networks based on binary Restricted Boltzmann Machines (RBM), built upon NumPy, TensorFlow and scikit-learn:
> Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.

> Fischer, Asja, and Christian Igel. "Training restricted Boltzmann machines: an introduction." Pattern Recognition 47.1 (2014): 25-39.

## Overview
This project works on Python 3.6 and follows the [scikit-learn](http://scikit-learn.org) API guidelines. The code includes two implementations: one is built on top of TensorFlow while the other one just uses NumPy. To decide which one to use is as easy as importing the classes from the correct module: ```dbn.tensorflow``` for TensorFlow or  ```dbn``` for NumPy.
```python
import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
```

## Usage
Clone this repository:
    
    git clone https://github.com/albertbup/deep-belief-network.git

and go to the root folder:
    
    cd deep-belief-network
    
#### The docker way:
Build the docker image (you'll need to have [docker installed](https://docs.docker.com/get-docker/) in your system):

    docker build --tag albertbup/deep-belief-network:1.0.5 .

Cool, let's go inside the container and run an example:
    
    docker run --rm -it -v ${PWD}:/code albertbup/deep-belief-network:1.0.5 bash
    # Now within the container...
    python example_classification.py

#### The virtualenv way:
Create a [virtual environment](https://virtualenv.pypa.io/en/latest/index.html) for **Python 3.6** and [activate](https://virtualenv.pypa.io/en/latest/user_guide.html#activators) it.

Next, install requirements:
  
    pip install -r requirements.txt
    
Finally, run an example:
    
    # Now within the virtual environment...
    python example_classification.py
        
## Citing the code
BibTex reference format:

        @misc{DBNAlbert,
        title={A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility},
        url={https://github.com/albertbup/deep-belief-network},
        author={albertbup},
        year={2017}}

 
