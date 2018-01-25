"""
Real estate multi variable linear regression exercise using TensorFlow."
Usage: python3 ml_course_ex1.py <data-file> <config-file> <plot data: true/false>"

Author: Kari Marttila
Project: https://github.com/karimarttila/ml-exercises/
"""

import csv
import sys
import math
import tensorflow as tf
import matplotlib.pyplot as mplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import configparser
import my_logger


class RealEstateLinearRegression:
    """
    Multi variable linear regression class.
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
        areas = []
        rooms = []
        prices = []
        with open(datafile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                areas.append(float(row[0]))
                rooms.append(float(row[1]))
                prices.append(float(row[2]))
        self.logger.debug("EXIT readCsvFile")
        return (areas, rooms, prices)


    def plotData(self, data):
        self.logger.debug("ENTER plotData")
        (areas, rooms, prices) = data
        areas_v = np.asarray(areas)
        rooms_v = np.asarray(rooms)
        prices_v = np.asarray(prices)
        fig = mplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(areas_v, rooms_v, prices_v, c='b', marker='o')
        ax.set_xlabel('area')
        ax.set_ylabel('rooms')
        ax.set_zlabel('price')
        mplot.show()
        return

    def plotJ_history(self, J_history, iterations):
        self.logger.debug("ENTER plotJ_history")
        mplot.plot(range(len(J_history)),J_history)
        mplot.axis([0,iterations,0,np.max(J_history)])
        mplot.show()
        self.logger.debug("EXIT plotJ_history")
        return


    def create_X_train(self, data):
        """ Create X_train with three columns: [1 (bias), area, rooms]."""
        self.logger.debug("ENTER create_X_train")
        (area, rooms, price) = data
        area_vec = np.asarray(area)
        area_mat = area_vec.reshape((area_vec.shape[0], 1))
        rooms_vec = np.asarray(rooms)
        rooms_mat = rooms_vec.reshape((rooms_vec.shape[0], 1))
        X_train = np.c_[area_mat,rooms_mat]
        self.logger.debug("EXIT create_X_train")
        return X_train


    def runLinearRegression(self, data):
        """
        Linear regression ex1 exercise.
        NOTE: In ex1a we didn't normalize the data. In this exercise ex1b
        we are normalizing data as it was normalized in the original
        ex1_multi exercise.
        """
        self.logger.debug("ENTER runLinearRegression")
        (area, rooms, price) = data
        iterations = int(self.config['DEFAULT']['iterations'])
        alpha = float(self.config['DEFAULT']['alpha'])  # Learning rate.
        self.logger.debug("Iterations: {0}".format(iterations))
        self.logger.debug("Alpha: {0}".format(alpha))
        X_train = self.create_X_train(data)
        X_train_means = X_train.mean(0)
        X__train_stds = X_train.std(0)
        X_train_normalized = (X_train - X_train_means) / X__train_stds
        X_train_normalized_bias = np.c_[np.ones(X_train_normalized.shape[0]),X_train_normalized]
        y_train_raw = np.asanyarray(price)
        y_train = y_train_raw.reshape((y_train_raw.shape[0], 1)) # Convert to (m,1)
        m = X_train_normalized_bias.shape[0]  # m samples
        n = X_train_normalized_bias.shape[1]  # n features (3: bias + area + rooms)
        X = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 1])
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
            sess.run(step,feed_dict={X:X_train_normalized_bias, y:y_train})
            J_history = np.append(J_history,sess.run(J,feed_dict={X:X_train_normalized_bias,y:y_train}))
        if (self.plotting_enabled):
            self.plotJ_history(J_history, iterations)

        # Make a test of 1650 area apartment with 3 rooms (first element is bias).
        X_test = np.asarray([[1650.0, 3.0]])
        X_test_normalized = (X_test - X_train_means) / X__train_stds
        X_test_normalized_bias = np.c_[np.ones(X_test_normalized.shape[0]),X_test_normalized]

        y_predicted = sess.run(y_prediction, feed_dict={X: X_test_normalized_bias})
        original_results_raw = np.asanyarray([289314.620338])
        original_results = original_results_raw.reshape((original_results_raw.shape[0], 1))
        deltas = y_predicted - original_results
        delta_percentages = (100*(y_predicted - original_results)/original_results)
        self.logger.info("Comparing to original ex1b predictions using area {0} and room {1}".format(
            X_test[0][0], X_test[0][1]))
        self.logger.info("Our prediction: {0:.2f} (original: {1:.2f}), delta: {2:.2f} ({3:.2f}%)".format(
            y_predicted[0][0], original_results[0][0], deltas[0][0], delta_percentages[0][0]))

        final_weights = sess.run(W)
        original_weights = [334302.063993, 100087.116006, 3673.548451] # Original weights (theta) in ex1b.m
        self.logger.info("Final trained weights: {0:.4f} (original: {1:.4f}), {2:.4f} (original: {3:.4f}), {4:.4f} (original: {5:.4f})".format(
            final_weights[0][0], original_weights[0], final_weights[1][0], original_weights[1], final_weights[2][0], original_weights[2]))

        original_J_history = [64300749594.565956, 2108850058.400706]
        self.logger.info("Convergance: J[1]: {0:.2f} (original: {1:.2f}), J[400]: {2:.2f} (original: {3:.2f})".format(
               J_history[1], original_J_history[0], J_history[400], original_J_history[1]))


    def run(self, datafile):
        self.logger.debug("ENTER run")
        self.logger.debug("data file: {0}".format(datafile))
        data = self.readCsvFile(datafile)
        (areas, rooms, prices) = data
        dataCount = len(areas)
        self.logger.debug("Read items: {0}".format(dataCount))
        if (self.plotting_enabled):
            self.plotData(data)
        ret = self.runLinearRegression(data)
        self.logger.debug("EXIT run")
        return ret


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
    model = RealEstateLinearRegression(config_file, plotting_enabled)
    return_code = model.run(data_file)
    sys.exit(return_code)
