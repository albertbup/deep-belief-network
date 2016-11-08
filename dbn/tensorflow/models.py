import numpy as np
import tensorflow as tf

from ..models import BinaryRBM as BaseBinaryRBM


class BinaryRBM(BaseBinaryRBM):
    """
    This class implements a Binary Restricted Boltzmann machine based on TensorFlow.
    """

    def fit(self, X):
        """
        Fit a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        self.n_visible_units = X.shape[1]

        # Initialize RBM parameters
        self._build_model()

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        if self.optimization_algorithm == 'sgd':
            self._stochastic_gradient_descent(X)
        else:
            raise ValueError("Invalid optimization algorithm.")
        self.sess.close()
        return self

    def _build_model(self):
        """
        Builds TensorFlow model.
        :return:
        """
        # weights and biases
        if self.activation_function == 'sigmoid':
            self.W = tf.Variable(tf.random_normal([self.n_hidden_units, self.n_visible_units]))
            self.c = tf.Variable(tf.random_normal([self.n_hidden_units]))
            self.b = tf.Variable(tf.random_normal([self.n_visible_units]))
            self._activation_function_class = tf.nn.sigmoid
        elif self.activation_function == 'relu':
            stddev = 0.1 / np.sqrt(self.n_visible_units)
            self.W = tf.Variable(
                tf.truncated_normal([self.n_hidden_units, self.n_visible_units], stddev=stddev, dtype=tf.float32))
            self.c = tf.Variable(tf.constant(stddev, shape=[self.n_hidden_units], dtype=tf.float32))
            self.b = tf.Variable(tf.constant(stddev, shape=[self.n_visible_units], dtype=tf.float32))
            self._activation_function_class = tf.nn.relu
        else:
            raise ValueError("Invalid activation function.")

        # TensorFlow operations
        self.visible_units_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_visible_units])
        self.compute_hidden_units_op = self._activation_function_class(
            tf.transpose(tf.matmul(self.W, tf.transpose(self.visible_units_placeholder))) + self.c)
        self.hidden_units_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_hidden_units])
        self.compute_visible_units_op = self._activation_function_class(
            tf.matmul(self.hidden_units_placeholder, self.W) + self.b)
        self.random_uniform_values = tf.Variable(tf.random_uniform([self.batch_size, self.n_hidden_units]))
        sample_hidden_units_op = tf.to_float(self.random_uniform_values < self.compute_hidden_units_op)

        # Positive gradient
        # Outer product. N is the batch size length.
        # From http://stackoverflow.com/questions/35213787/tensorflow-batch-outer-product
        positive_gradient_op = tf.batch_matmul(tf.expand_dims(sample_hidden_units_op, 2),  # [N, U, 1]
                                               tf.expand_dims(self.visible_units_placeholder, 1))  # [N, 1, V]

        # Negative gradient
        # Gibbs sampling
        sample_hidden_units_gibbs_step_op = sample_hidden_units_op
        for t in range(self.contrastive_divergence_iter):
            compute_visible_units_op = self._activation_function_class(
                tf.matmul(sample_hidden_units_gibbs_step_op, self.W) + self.b)
            compute_hidden_units_gibbs_step_op = self._activation_function_class(
                tf.transpose(tf.matmul(self.W, tf.transpose(compute_visible_units_op))) + self.c)
            sample_hidden_units_gibbs_step_op = tf.to_float(tf.Variable(
                tf.random_uniform([self.batch_size, self.n_hidden_units])) < compute_hidden_units_gibbs_step_op)

        negative_gradient_op = tf.batch_matmul(tf.expand_dims(sample_hidden_units_gibbs_step_op, 2),  # [N, U, 1]
                                               tf.expand_dims(compute_visible_units_op, 1))  # [N, 1, V]

        compute_delta_W = tf.reduce_mean(positive_gradient_op - negative_gradient_op, 0)
        compute_delta_b = tf.reduce_mean(self.visible_units_placeholder - compute_visible_units_op, 0)
        compute_delta_c = tf.reduce_mean(sample_hidden_units_op - sample_hidden_units_gibbs_step_op, 0)

        self.update_W = tf.assign_add(self.W, self.learning_rate * compute_delta_W)
        self.update_b = tf.assign_add(self.b, self.learning_rate * compute_delta_b)
        self.update_c = tf.assign_add(self.c, self.learning_rate * compute_delta_c)

    def _stochastic_gradient_descent(self, _data):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :return:
        """
        for iteration in range(1, self.n_epochs + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
            for batch in self.get_batches(self.batch_size, data):
                if len(batch) < self.batch_size:
                    # Pad with zeros
                    pad = np.zeros((self.batch_size - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
                    batch = np.vstack((batch, pad))
                self.sess.run(tf.initialize_variables(
                    [self.random_uniform_values]))  # Need to re-sample from uniform distribution
                self.sess.run([self.update_W, self.update_b, self.update_c],
                              feed_dict={self.visible_units_placeholder: batch})
            if self.verbose:
                error = self._compute_reconstruction_error(data)
                print ">> Epoch %d finished \tRBM Reconstruction error %f" % (iteration, error)

    def _compute_hidden_units_matrix(self, matrix_visible_units):
        """
        Computes hidden unit outputs.
        :param matrix_visible_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return self.sess.run(self.compute_hidden_units_op,
                             feed_dict={self.visible_units_placeholder: matrix_visible_units})

    def _compute_visible_units_matrix(self, matrix_hidden_units):
        """
        Computes visible (or input) unit outputs.
        :param matrix_hidden_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return self.sess.run(self.compute_visible_units_op,
                             feed_dict={self.hidden_units_placeholder: matrix_hidden_units})
