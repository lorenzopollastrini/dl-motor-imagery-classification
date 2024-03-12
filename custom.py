import logging

import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.initializers import Constant, TruncatedNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

import graph_utils

tf.get_logger().setLevel(logging.ERROR)


@register_keras_serializable(name='GraphConv')
class GraphConv(Layer):

    def __init__(self, laplacian, n_output_filters, k):
        super(GraphConv, self).__init__()
        self.laplacian = laplacian
        self.n_output_filters = n_output_filters
        self.k = k

    def build(self, input_shape):
        n_nodes = input_shape[1]
        n_input_filters = input_shape[-1]
        self.weights_matrix = self.add_weight(
            shape=(n_input_filters * (self.k + 1), self.n_output_filters),
            initializer=TruncatedNormal(mean=0.0, stddev=0.1),
            trainable=True
        )
        self.biases_matrix = self.add_weight(
            shape=(1, n_nodes, self.n_output_filters),
            initializer=Constant(0.1),
            trainable=True
        )

    def call(self, inputs):
        n_samples = tf.shape(inputs)[0]  # N
        n_nodes = tf.shape(inputs)[1]  # M
        n_input_filters = inputs.shape[-1]  # F_in

        # Normalize Laplacian and store it as a TensorFlow sparse tensor
        laplacian = csr_matrix(self.laplacian)
        laplacian = graph_utils.normalize_laplacian(laplacian, max_eigenvalue=2)
        laplacian = laplacian.tocoo()
        indices = np.column_stack((laplacian.row, laplacian.col))
        laplacian = tf.SparseTensor(indices, tf.cast(laplacian.data, tf.float32), laplacian.shape)
        laplacian = tf.sparse.reorder(laplacian)

        # Chebyshev basis
        x_0 = tf.transpose(a=inputs, perm=[1, 2, 0])  # Shape: M × F_in × N
        x_0 = tf.reshape(x_0, [n_nodes, n_input_filters * n_samples])  # Shape: M × (F_in * N)
        x = tf.expand_dims(x_0, 0)  # Shape: 1 × M × (F_in * N)

        if self.k > 1:
            x_1 = tf.sparse.sparse_dense_matmul(laplacian, x_0)
            x = tf.concat([x, tf.expand_dims(x_1, 0)], axis=0)
            for k in range(2, self.k + 1):
                x_2 = 2 * tf.sparse.sparse_dense_matmul(laplacian, x_1) - x_0  # Shape: M × (F_in * N)
                x = tf.concat([x, tf.expand_dims(x_2, 0)], axis=0)
                # x.shape at the end of the loop: k × M × (F_in * N)
                x_0, x_1 = x_1, x_2

        x = tf.reshape(x, [self.k + 1, n_nodes, n_input_filters, n_samples])  # Shape: k × M × F_in × N
        x = tf.transpose(a=x, perm=[3, 1, 2, 0])  # Shape: N × M × F_in × k
        x = tf.reshape(x, [n_samples * n_nodes, n_input_filters * (self.k + 1)])  # Shape: (N * M) × (F_in * k)

        # W.shape: (F_in * k) × F_out
        # b: 1 × M × F_out
        x = tf.matmul(x, self.weights_matrix)  # (N * M) × F_out
        x = tf.reshape(x, [n_samples, n_nodes, self.n_output_filters])  # N × M × F_out
        x = x + self.biases_matrix
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_output_filters

    def get_config(self):
        config = super().get_config()
        config.update(
            {'laplacian_data': self.laplacian.data,
             'laplacian_indices': self.laplacian.indices,
             'laplacian_indptr': self.laplacian.indptr,
             'laplacian_shape': self.laplacian.shape,
             'n_output_filters': self.n_output_filters,
             'k': self.k}
        )
        return config

    @classmethod
    def from_config(cls, config):
        laplacian_data = config.pop('laplacian_data')['config']['value']
        laplacian_indices = config.pop('laplacian_indices')['config']['value']
        laplacian_indices = np.asarray(laplacian_indices)
        laplacian_indptr = config.pop('laplacian_indptr')['config']['value']
        laplacian_indptr = np.asarray(laplacian_indptr)
        laplacian_shape = config.pop('laplacian_shape')

        laplacian = csr_matrix((laplacian_data, laplacian_indices, laplacian_indptr), shape=laplacian_shape)

        n_output_filters = config.pop('n_output_filters')
        k = config.pop('k')

        return cls(laplacian, n_output_filters, k)
