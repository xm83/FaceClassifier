#!/usr/bin/env python
# Script to train and test a neural network with TF's Keras API for face detection

import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf

def load_data_from_npz_file(file_path):
    """
    Load data from npz file
    :param file_path: path to npz file with training data
    :return: input features and target data as numpy arrays
    """
    data = np.load(file_path)
    return data['input'], data['target']


def normalize_data_per_row(data):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :return: normalized data with pixel values in [0,1]
    """

    # sanity checks!
    assert len(data.shape) == 4, "Expected the input data to be a 4D matrix"
    
    pixels = np.asarray(data)
    # print('Data Type: %s' % pixels.dtype)
    # print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    # print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

    return pixels


def train_model(model, train_input, train_target, val_input, val_target, logs_dir, 
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])

    # tensorboard callback
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'weights.h5'),
                                                            monitor='val_loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            period=1)

    # do training for the specified number of epochs and with the given batch size
    model.fit(train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(val_input, val_target),
             callbacks=[tbCallBack, checkpointCallBack])


def split_data(input, target, train_percent):
    """
    Split the input and target data into two sets
    :param input: inputs [Nx2] matrix
    :param target: target [Nx1] matrix
    :param train_percent: percentage of the data that should be assigned to training
    :return: train_input, train_target, test_input, test_target
    """
    assert input.shape[0] == target.shape[0], \
        "Number of inputs and targets do not match ({} vs {})".format(input.shape[0], target.shape[0])

    indices = range(input.shape[0])
    np.random.shuffle(indices)

    num_train = int(input.shape[0]*train_percent)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices,:], input[test_indices,:], target[test_indices,:]


def main(npz_data_file, batch_size, epochs, lr, val, logs_dir):
    """
    Main function that performs training and test on a validation set
    :param npz_data_file: npz input file with training data
    :param batch_size: batch size to use at training time
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param val: percentage of the training data to use as validation
    :param logs_dir: directory where to save logs and trained parameters/weights
    """

    input, target = load_data_from_npz_file(npz_data_file)
    N = input.shape[0]
    assert N == target.shape[0], \
        "The input and target arrays had different amounts of data ({} vs {})".format(N, target.shape[0]) # sanity check!
    print "Loaded {} training examples.".format(N)

    # TODO. Complete. Implement code to train a network for image classification
    H = input.shape[1]
    W = input.shape[2]
    C = input.shape[3]

    # split training data into actual training and validation
    train_input, train_target, val_input, val_target = split_data(input, target, 0.8)

    # normalize training data
    train_input = normalize_data_per_row(train_input)
    val_input = normalize_data_per_row(val_input)
    
    # build a CNN model: based on VGGnet
    # https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
    model = tf.keras.models.Sequential()
    # filter size = 32 = size of the output dim i.e. num of output filters in the convolution
    # kernel size = (3,3) = height and width of the 2d convolution window
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape = (H, W, C), activation='relu'))
    # reduce the spatial size of the incoming features and thus reduce overfitting
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Flatten())
    # randomly drops some layers then learns w reduced network; not rely on a single layer; helps avoid overfitting; 0.5 = randomly drop half of the layers
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    # use sigmoid to return a prob dist in range of 0 and 1
    # or 'softmax'
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name="face_model"))

     # train the model
    print "\n\nTRAINING..."
    train_model(model, train_input, train_target, val_input, val_target, logs_dir, 
                epochs=epochs, learning_rate=lr, batch_size=batch_size)


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=50)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=100)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=1e-3)
    parser.add_argument("--val", help="percent of training data to use for validation",
                        type=float, default=0.8)
    parser.add_argument("--input", help="input file (npz format)",
                        type=str, required=True)
    parser.add_argument("--logs_dir", help="logs directory",
                        type=str, default="")
    args = parser.parse_args()

    if len(args.logs_dir) == 0: # parameter was not specified
        args.logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))

    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir)

    # run the main function
    main(args.input, args.batch_size, args.epochs, args.lr, args.val, args.logs_dir)
    sys.exit(0)
