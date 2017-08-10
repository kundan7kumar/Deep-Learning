#!/usr/bin/env python

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from prepare_data import get_training_data, get_testing_data

import sys
import os
import gzip
import pickle
import numpy

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'


def load_data():
    """Get data with labels, split into training, validation and test set."""
    ourtrainingdata = get_training_data()
    ourtestingdata = get_testing_data()

    #X_train, y_train = data[0]
    X_train = ourtrainingdata[0]
    y_train = ourtrainingdata[1]

    #X_test, y_test = data[2]
    X_test = ourtestingdata[0]
    y_test = ourtestingdata[1]

    y_train = numpy.asarray(y_train, dtype=numpy.int32)
    y_test = numpy.asarray(y_test, dtype=numpy.int32)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=len(X_train),
        num_examples_test=len(X_test),
        input_dim=X_train[0],
        output_dim=10,
    )


def nn_example(data):
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 28*28),
        hidden_num_units=100,  # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=30,
        verbose=1,
        )

    # Train the network
    net1.fit(numpy.array(data['X_train']), numpy.array(data['y_train']))

    # Try the network on new data
    # print("Feature vector (100-110): %s" % data['X_test'][0][100:110])
    print("Actual Label: %s" % str(data['y_test'][9000]))
    print("Predicted: %s" % str(net1.predict([data['X_test'][9000]])))

    preds = net1.predict(data['X_test'])

    cm = confusion_matrix(data['y_test'], preds)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    data = load_data()
    print("Got %i testing datasets." % len(data['X_train']))
    nn_example(data)

if __name__ == '__main__':
    main()