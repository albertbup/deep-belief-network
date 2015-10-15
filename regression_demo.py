from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.models import SupervisedDBNRegression


# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
classifier = SupervisedDBNRegression(hidden_layers_structure=[200], learning_rate_rbm=0.01, learning_rate=0.001,
                                     max_epochs_rbm=100, max_iter_backprop=1000, lambda_param=0.0)
classifier.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = classifier.predict(X_test)
print 'Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred))


