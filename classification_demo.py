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
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.01,
                                         learning_rate=0.001,
                                         n_epochs_rbm=30,
                                         n_iter_backprop=1000,
                                         l2_regularization=0.0)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)
print 'Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)

