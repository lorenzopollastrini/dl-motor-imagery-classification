import numpy as np
import pandas as pd


def load_dataset(dataset_directory):
    training_set = pd.read_csv(dataset_directory + '/' + 'training_set.csv', header=None)
    training_set = np.array(training_set).astype('float32')

    training_labels = pd.read_csv(dataset_directory + '/' + 'training_labels.csv', header=None)
    training_labels = np.array(training_labels).astype('float32')
    training_labels = np.squeeze(training_labels)

    test_set = pd.read_csv(dataset_directory + '/' + 'test_set.csv', header=None)
    test_set = np.array(test_set).astype('float32')

    test_labels = pd.read_csv(dataset_directory + '/' + 'test_labels.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    test_labels = np.squeeze(test_labels)

    return training_set, training_labels, test_set, test_labels
