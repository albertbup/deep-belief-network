from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np

from RBM import RBM
from DBN import DBN


def plot_scatter_color(X, y):
    colors = cm.rainbow(np.linspace(0, 1, np.unique(y).size))
    for label, color in zip(np.unique(y), colors):
        plt.scatter(X[y == label, 0], X[y == label, 1], c=color)

# Get data
mnist = fetch_mldata('MNIST original')
XX = mnist.data
# print y
#digits = load_digits()
#XX = digits.data

XX /= 255.0
X = XX[60000:]
y = mnist.target[60000:]
# X = (X - np.min(X)) / (np.max(X) - np.min(X))

# # Do LDA
# lda = LDA(n_components=2)
# X_LDA = lda.fit_transform(X, y)
#
# fig = plt.figure()
# plot_scatter_color(X_LDA, y)
# plt.title('FDA with all features')
# fig.tight_layout()
# plt.savefig('LDA.png', format='png')
#
# # Train RBM
# rbm = RBM(2)
# rbm.fit(X, learning_rate=0.3, epochs=200)
#
# X_RBM = rbm.transform(X)
#
# fig = plt.figure()
# plot_scatter_color(X_RBM, y)
# plt.title('RBM with all features')
# fig.tight_layout()
# plt.savefig('RBM.png', format='png')

# # Do PCA
# pca = PCA(n_components=60)
# X_PCA = pca.fit_transform(X)

# Train DBN
dbn = DBN([100, 100, 500], learning_rate=0.3, max_iter_backprop=10, max_epochs_rbm=1, lambda_param=0.0)
dbn.fit(X, labels=y)

X_t = XX[:60000]
y_t = mnist.target[:60000]

X_DBN = dbn.transform(X_t)
Y_DBN = dbn.predict(X_t)

# Do LDA
lda = LDA(n_components=2)
X_LDA = lda.fit_transform(X_DBN, y_t)

fig = plt.figure()
plot_scatter_color(X_LDA, y_t)
plt.title('DBM on MNIST')
fig.tight_layout()
plt.savefig('DBM.png', format='png')
