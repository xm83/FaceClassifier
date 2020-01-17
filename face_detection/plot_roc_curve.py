#!/usr/bin/env python
# Script to train and test a neural network with TF's Keras API for face detection

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train_face_detection import normalize_data_per_row, load_data_from_npz_file

def main(input_file, weights_file):
    """
    Evaluate the model on the given input data
    :param input_file: npz datafile
    :param weights_file: h5 file with model definition and weights
    :param norm_file: normalization params
    """
    # load data
    input, target = load_data_from_npz_file(input_file)
    N = input.shape[0]
    assert N == target.shape[0], \
        "The input and target arrays different amounts of data ({} vs {})".format(N,
                                                                                      target.shape[0])  # sanity check!
    print "Loaded {} testing examples.".format(N)

    # normalize the inputs
    norm_input = normalize_data_per_row(input)

    # load keras model from file
    model = tf.keras.models.load_model(weights_file)

    # set loss for evaluation
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy']) # accuracy threshold is 0.5

    # output model predictions on "prob" variable
    prob = model.predict(norm_input)

     # generate roc thresholds
    thresholds = [x/100.0 for x in range(0,100,2)]
    
    tpr = [] # list of true positive rate per threshold
    fpr = [] # list of false positive rate per threshold
  
    # compute the true positive rate and the false positive rate for each of the thresholds
    for t in thresholds:
  
        # turn predicted probabilities to 0-1 values based on the threshold
        prediction = np.zeros(prob.shape)
        prediction[prob > t] = 1
        # compute tpr and fpr based on the predictions and the target values from the dataset     
        tp = 0 # num of true pos
        fp = 0 # num of false pos
        p = 0 # num of pos
        n = 0 # num of neg
        for i in range(len(target)):
            if target[i] == 1:
                p = p + 1
                if prediction[i] == 1:
                    tp = tp + 1
            if target[i] == 0:
                n = n + 1
                if prediction[i] == 1:
                    fp = fp + 1
        current_tpr = float(tp) / p
        current_fpr = float(fp) / n
        print(tp, fp, current_tpr, current_fpr, p, n)

        tpr.append(current_tpr)
        fpr.append(current_fpr)
    print(tpr, fpr)
    # pick threshold that minimizes l2 distance to top-left corner of the graph (fpr = 0, tpr = 1)
    # index of the threshold for which (fpr, tpr) get closest to (0,1) in the Euclidean sense
    min_dist = 3 # worst case is 1 + 1 = 2
    index = 0
    for i in range(len(thresholds)):
        current_fpr = fpr[i]
        current_tpr = tpr[i]
        dist = current_fpr**2 + (1 - current_tpr)**2
        if dist < min_dist:
            min_dist = dist
            index = i

    print "Best threshold was: {} (TPR = {}, FPR = {})".format(thresholds[index], tpr[index], fpr[index])
    #Best threshold was: 0.4 (TPR = 0.981141199226, FPR = 0.0168067226891)
    
    # plot the ROC curve with matplotlib (remember to import it as "import matplotlib.pyplot as plt")
    plt.plot(fpr, tpr)
    plt.scatter(fpr[index], tpr[index], s=20, c='r')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('ROC Curve')
    plt.show()


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file (npz format)",
                        type=str, required=True)
    parser.add_argument("--logs_dir", help="logs directory",
                        type=str, required=True)
    parser.add_argument("--weights_filename", help="name for the weights file",
                        type=str, default="weights.h5")
    args = parser.parse_args()

    weights_path = os.path.join(args.logs_dir, args.weights_filename)
    
    # run the main function
    main(args.input, weights_path)
    sys.exit(0)
