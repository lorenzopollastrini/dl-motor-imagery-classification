import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Flatten, MaxPooling2D, Reshape
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.regularizers import L2

from custom import GraphConv
from data_loader import load_dataset
import graph_utils
import pooling

tf.get_logger().setLevel(logging.ERROR)

n_classes = 4

dataset_directory = 'dataset'

training_set, training_labels, test_set, test_labels = load_dataset(dataset_directory=dataset_directory)

training_labels = tf.one_hot(training_labels, n_classes)
test_labels = tf.one_hot(test_labels, n_classes)

adjacency_matrix = pd.read_csv(dataset_directory + '/' + 'adjacency_matrix.csv', header=None)
adjacency_matrix = np.array(adjacency_matrix).astype('float32')
adjacency_matrix = csr_matrix(adjacency_matrix)

graphs, permutations = pooling.coarsen(adjacency_matrix, levels=5, self_connections=False)
training_set = pooling.permute_data(training_set, permutations)
test_set = pooling.permute_data(test_set, permutations)

laplacians = [graph_utils.laplacian(adjacency_matrix, normalized=True) for adjacency_matrix in graphs]

model = Sequential(
    [
        Input(shape=(64,)),
        Reshape((64, 1)),

        GraphConv(laplacians[0], 16, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((64, 16, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        Reshape((32, 16)),

        GraphConv(laplacians[1], 32, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((32, 32, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        Reshape((16, 32)),

        GraphConv(laplacians[2], 64, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((16, 64, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        Reshape((8, 64)),

        GraphConv(laplacians[3], 128, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((8, 128, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        Reshape((4, 128)),

        GraphConv(laplacians[4], 256, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((4, 256, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        Reshape((2, 256)),

        GraphConv(laplacians[5], 512, 2),
        BatchNormalization(),
        Activation('softplus'),
        Reshape((2, 512, 1)),
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'),

        Flatten(),
        Dense(units=n_classes, activation='softmax', kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001)),
    ]
)

model.summary()

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(),
    metrics=['categorical_accuracy', F1Score(average='macro')]
)

model_checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/checkpoint.keras',
    monitor='val_f1_score',
    verbose=1,
    save_best_only=True,
    mode='max'
)

model.fit(
    training_set, training_labels,
    batch_size=1024, epochs=100,
    validation_data=(test_set, test_labels),
    callbacks=[model_checkpoint_callback]
)
