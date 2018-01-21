"""
Profit/Population linear regression exercise using TensorFlow."
Usage: python3 ml_course_ex1.py <data-file> <config-file> <plot data: true/false>"

Author: Kari Marttila
Project: https://github.com/karimarttila/ml-exercises/
"""

import os
import csv
import sys
import math
import time
import logging.config
import tensorflow as tf
import matplotlib.pyplot as mplot
import numpy as np
import configparser
import my_logger


class ProfitPopulationLinearRegression:
    """
    Linear regression class.
    """
    config_file = None
    logger = None
    plotting_enabled = False
    config = None
    random_generator = np.random

    def __init__(self, config_file, plotting_enabled):
        self.plotting_enabled = plotting_enabled
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        log_level = self.config['DEFAULT']['log_level']
        self.logger = my_logger.FileConfigLogger("TF", None, log_level, "LOG_CFG")


    def readCsvFile(self, datafile):
        self.logger.debug("ENTER readCsvFile")
        populations = []
        profits = []
        with open(datafile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                populations.append(float(row[0]))
                profits.append(float(row[1]))
        self.logger.debug("EXIT readCsvFile")
        return (populations, profits)


    def plotData(self, data):
        self.logger.debug("ENTER plotData")
        (populations, profits) = data
        mplot.plot(populations, profits, "bx")
        yint = range(math.ceil(min(profits))-1, math.ceil(max(profits))+1)
        xint = range(math.ceil(min(populations))-1, math.ceil(max(populations))+1)
        mplot.yticks(yint)
        mplot.xticks(xint)
        mplot.xlabel("Population")
        mplot.ylabel("Profit")
        mplot.show()
        self.logger.debug("EXIT plotData")
        return

    def plotJ_history(self, J_history, iterations):
        self.logger.debug("ENTER plotJ_history")
        mplot.plot(range(len(J_history)),J_history)
        mplot.axis([0,iterations,0,np.max(J_history)])
        mplot.show()
        self.logger.debug("EXIT plotJ_history")
        return


    def appendBias(self, vec):
        self.logger.debug("ENTER appendBias")
        # See: https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
        # Reshape the vector vec (m) to (m,1), i.e. [1.34, 2.54] => [[1.34],[2.54]]
        matrix = vec.reshape((vec.shape[0], 1)) # New matrix
        # Add bias as first column.
        m = matrix.shape[0] # Number of rows, i.e. sample count.
        n = matrix.shape[1] # Number of features.
        # Add bias (1) => (m,2), i.e. => [[1.0, 1.34],[1,0, 2.54]]
        withBias = np.reshape(np.c_[np.ones(m),matrix],[m,n + 1])
        self.logger.debug("EXIT appendBias")
        return withBias

    def runLinearRegression(self, data):
        """
        Linear regression ex1 exercise.
        NOTE: We are NOT normalizing data since it was not normalized
        in the original ex1 either (you should normalize data when
        using linear regression).
        """
        self.logger.debug("ENTER runLinearRegression")
        (populations, profits) = data
        iterations = int(self.config['DEFAULT']['iterations'])
        alpha = float(self.config['DEFAULT']['alpha'])  # Learning rate.
        self.logger.debug("Iterations: {0}".format(iterations))
        self.logger.debug("Alpha: {0}".format(alpha))
        X_train = np.asarray(populations) # Features, i.e. population
        y_train = np.asarray(profits)     # Labels, i.e. profits

        # Add bias to X.
        X_train_bias = self.appendBias(X_train)
        # Reshape y to matrix.
        y_train_m = y_train.reshape((y_train.shape[0], 1)) # Convert to (m,1)

        m = X_train_bias.shape[0]  # m samples
        n = X_train_bias.shape[1]  # n features (2: bias + population)
        X = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 1])

        # Theta (in ML course terminology) or weights (TensorFlow terminology).
        W = tf.Variable(tf.ones([n,1]), name="weights")
        init = tf.global_variables_initializer()
        y_prediction = tf.matmul(X, W) # As matrix multiplication in ex1.
        # Cost
        J = (1 / (2 * m)) * tf.reduce_sum(tf.pow(y_prediction - y, 2))
        step = tf.train.GradientDescentOptimizer(alpha).minimize(J)
        sess = tf.Session()
        sess.run(init)

        # For recording cost history.
        J_history = np.empty(shape=[1],dtype=float)
        # Train iterations.
        for i in range(iterations):
            sess.run(step,feed_dict={X:X_train_bias, y:y_train_m})
            J_history = np.append(J_history,sess.run(J,feed_dict={X:X_train_bias,y:y_train_m}))
        if (self.plotting_enabled):
            self.plotJ_history(J_history, iterations)

        # Make a couple of predictions to compare to original ex1.
        X_test = np.asarray([3.5, 7.0])
        X_test_bias = self.appendBias(X_test)
        y_predicted = sess.run(y_prediction, feed_dict={X: X_test_bias})
        original_results_raw = np.asanyarray([0.451977, 4.534245])
        original_results = original_results_raw.reshape((original_results_raw.shape[0], 1))
        deltas = y_predicted - original_results
        delta_percentages = (100*(y_predicted - original_results)/original_results)
        self.logger.info("Comparing to original ex1 predictions using populations 3.5 and 7.0")
        self.logger.info("Population: {0}, profits: our predicion: {1:.6f} (original: {2:.6f}), delta: {3:.2f} ({4:.2f}%)".format(
            X_test[0], y_predicted[0][0], original_results[0][0], deltas[0][0], delta_percentages[0][0]))
        self.logger.info("Population: {0}, profits: our predicion: {1:.6f} (original: {2:.6f}), delta: {3:.2f} ({4:.2f}%)".format(
            X_test[1], y_predicted[1][0], original_results[1][0], deltas[1][0], delta_percentages[1][0]))

        final_weights = sess.run(W)
        original_weights = [-3.6303, 1.1664] # Original weights (theta) in ex1.m
        self.logger.info("Final trained weights: {0:.4f} (original: {1:.4f}), {2:.4f} (original: {3:.4f})".format(
            final_weights[0][0], original_weights[0], final_weights[1][0], original_weights[1]))

        original_J_history = [6.7372, 4.4834]
        self.logger.info("Convergance: J[1]: {0:.4f} (original: {1:.4f}), J[1500]: {2:.4f} (original: {3:.4f})".format(
               J_history[1], original_J_history[0], J_history[1500], original_J_history[1]))

        # Close TF session.
        sess.close()
        self.logger.debug("EXIT runLinearRegression")
        return


    def run(self, datafile):
        self.logger.debug("ENTER run")
        self.logger.debug("data file: {0}".format(datafile))
        (populations, profits) = self.readCsvFile(datafile)
        dataCount = len(populations)
        self.logger.debug("Read items: {0}".format(dataCount))
        if (self.plotting_enabled):
            self.plotData((populations,profits))
        self.runLinearRegression((populations,profits))
        self.logger.debug("EXIT run")
        return


def showError():
    print(__doc__)
    return


def getArguments(argv):
    if len(argv) != 4:
        showError()
        return -1
    data_file = argv[1]
    config_file = argv[2]
    plotting_enabled = (argv[3] == "true")
    return(data_file, config_file, plotting_enabled)


if __name__ =='__main__':
    (data_file, config_file, plotting_enabled) = getArguments(sys.argv)
    model = ProfitPopulationLinearRegression(config_file, plotting_enabled)
    return_code = model.run(data_file)
    sys.exit(return_code)
