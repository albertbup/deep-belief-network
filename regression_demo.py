import tensorflow as tf
from dbn.tensorflow import SupervisedDBNRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np

np.random.seed(1337)  # for reproducibility


# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

LEARNING_RATE_BASE = 0.01  # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
LEARNING_RATE_STEP = 100  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
# 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
gloabl_steps = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)
print(X_train.shape)
# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[80],
                                    learning_rate_rbm=0.01,
                                    learning_rate=learning_rate,
                                    n_epochs_rbm=1,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' %
      (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
