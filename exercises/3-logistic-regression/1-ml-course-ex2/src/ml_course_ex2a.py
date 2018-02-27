"""
University admission logistic regression using TensorFlow."
Usage: python3 ml_course_ex2a.py <data-file> <config-file> <plot data: true/false>"

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


class UniversityAdmissionLogisticRegression:
    """
    Logistic regression class.
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

    def read_csv_file(self, datafile):
        self.logger.debug("ENTER read_csv_file")
        exam1 = []
        exam2 = []
        admittance = []
        with open(datafile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                exam1.append(float(row[0]))
                exam2.append(float(row[1]))
                admittance.append(int(row[2]))
        self.logger.debug("EXIT read_csv_file")
        return (exam1, exam2, admittance)

    def plot_data(self, data, weights=None):
        self.logger.debug("ENTER plot_data")
        (exam1, exam2, admittance) = data
        exam1_v = np.asarray(exam1)
        exam2_v = np.asarray(exam2)
        admittance_v = np.asarray(admittance)
        admittance_passed_v = (admittance_v == 1)
        admittance_not_passed_v = (admittance_v == 0)
        exam1_passed_v = np.extract(admittance_passed_v, exam1_v)
        exam2_passed_v = np.extract(admittance_passed_v, exam2_v)
        exam1_not_passed_v = np.extract(admittance_not_passed_v, exam1_v)
        exam2_not_passed_v = np.extract(admittance_not_passed_v, exam2_v)

        fig = mplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Exam1 score')
        ax.set_ylabel('Exam2 score')
        major_ticks = np.arange(0, 101, 10)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.plot(exam1_passed_v, exam2_passed_v, "bx", label= 'Admitted')
        ax.plot(exam1_not_passed_v, exam2_not_passed_v, "ro", fillstyle='none', label= 'Not admitted')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)

        # Draw decision boundary as in ex2, plotDecisionBoundary.m function.
        if (weights is not None):
            plot_x = np.asarray([exam1_v.min() - 2, exam1_v.max() + 2])
            plot_y = (-1.0 / weights[2]) * (weights[1] * plot_x + weights[0])
            ax.plot(plot_x, plot_y, "g")

        mplot.show()
        self.logger.debug("EXIT plot_data")
        return


    def create_X_train(self, data):
        """ Create X_train with three columns: [exam1, exam2]."""
        self.logger.debug("ENTER create_X_train")
        (exam1, exam2, admittance) = data
        exam1_vec = np.asarray(exam1)
        exam1_mat = exam1_vec.reshape((exam1_vec.shape[0], 1))
        exam2_vec = np.asarray(exam2)
        exam2_mat = exam2_vec.reshape((exam2_vec.shape[0], 1))
        X_train = np.c_[exam1_mat, exam2_mat]
        self.logger.debug("EXIT create_X_train")
        return X_train


    def get_variables(self, data):
        self.logger.debug("ENTER get_variables")
        (exam1, exam2, admittance) = data
        X_train = self.create_X_train(data)
        X_train_bias = np.c_[np.ones(X_train.shape[0]),X_train]
        y_train_raw = np.asanyarray(admittance)
        y_train = y_train_raw.reshape((y_train_raw.shape[0], 1)) # Convert to (m,1)
        m = X_train_bias.shape[0]  # m samples
        n = X_train_bias.shape[1]  # n features (3: bias + exam1 + exam2)
        X = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 1])
        W = tf.Variable(tf.zeros([n,1]), name="weights")
        return (X_train_bias, y_train, X, y, W, n, m)


    def run_logistic_regression_plain_initial_cost(self, data, ml_config):
        """
        Logistic regression ex2 exercise, initial cost part.
        NOTE: In the original ML Coursera course the data was NOT normalized,
        so we don't normalize it in this exercise either.
        Using plain cost formula as we did in the original exercise using Octave.
        """
        self.logger.debug("ENTER run_logistic_regression_plain_initial_cost")
        (iterations, alpha) = ml_config
        (X_train_bias, y_train, X, y, W, n, m) = self.get_variables(data)
        # Using sigmoid as in the original exercise, but here using TF library function.
        h = tf.nn.sigmoid(tf.matmul(X,W)) # As in the original exercise: h = sigmoid(X * theta);
        # Cost function with plain formula as we did in the original exercise.
        # Original formula: J = (1 / m) * ( (((-y)') * (log(h))) - (((1 - y)') * (log(1-h))));
        J_plain_cost = tf.reduce_sum((1/m) *
                                     (tf.subtract(
                                        tf.multiply(-y, tf.log(h)),
                                         tf.multiply(1 - y, tf.log(1 - h)))) )
        init = tf.global_variables_initializer()
        # Using Python 'with' statement (context manager) this time - no need to close session explicitely later.
        with tf.Session() as session:
            session.run(init)
            J_value = session.run(J_plain_cost,feed_dict={X:X_train_bias,y:y_train})
            self.logger.debug("Comparing cost with the original cost of the Coursera exercise:")
            original_cost =  0.693147
            delta = J_value - original_cost
            delta_percentage = (100*(J_value - original_cost)/original_cost)
            self.logger.debug("J_value: {0:.6f}, original cost: {1:.6f}, delta: {2:.6f} ({3:.4f}%)".format(
                J_value, original_cost, delta, delta_percentage))

        self.logger.debug("EXIT run_logistic_regression_plain_initial_cost")
        return 0  # Everything ok.

    def run_logistic_regression_sigmoid_cross_entropy_initial_cost(self, data, ml_config):
        """
        Logistic regression ex2 exercise, initial cost part.
        NOTE: In the original ML Coursera course the data was NOT normalized,
        so we don't normalize it in this exercise either.
        Using sigmoid with cross entropy to compare to plain cost function method.
        """
        self.logger.debug("ENTER run_logistic_regression_sigmoid_cross_entropy_initial_cost")
        (iterations, alpha) = ml_config
        (X_train_bias, y_train, X, y, W, n, m) = self.get_variables(data)
        # Note: just multiplying the matrices since sigmoid applied in the cost function.
        h = tf.matmul(X,W)
        J_cross_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=y))
        init = tf.global_variables_initializer()
        # Using Python 'with' statement (context manager) this time - no need to close session explicitely later.
        with tf.Session() as session:
            session.run(init)
            J_value = session.run(J_cross_cost,feed_dict={X:X_train_bias,y:y_train})
            self.logger.debug("Comparing cost with the original cost of the Coursera exercise:")
            original_cost =  0.693147
            delta = J_value - original_cost
            delta_percentage = (100*(J_value - original_cost)/original_cost)
            self.logger.debug("J_value: {0:.6f}, original cost: {1:.6f}, delta: {2:.6f} ({3:.4f}%)".format(
                J_value, original_cost, delta, delta_percentage))

        self.logger.debug("EXIT run_logistic_regression_sigmoid_cross_entropy_initial_cost")
        return 0  # Everything ok.


    def plot_j_history(self, J_history, iterations):
        self.logger.debug("ENTER plot_j_history")
        mplot.plot(range(len(J_history)),J_history)
        mplot.axis([0,iterations,0,np.max(J_history)])
        mplot.show()
        self.logger.debug("EXIT plot_j_history")
        return


    def run_logistic_regression_plain_training_cost(self, data, ml_config):
        """
        Logistic regression ex2 exercise, training part.
        NOTE: In the original ML Coursera course the data was NOT normalized,
        so we don't normalize it in this exercise either.
        Using plain cost formula as we did in the original exercise using Octave.
        """
        self.logger.debug("ENTER run_logistic_regression_plain_training_cost")
        (iterations, alpha) = ml_config
        (X_train_bias, y_train, X, y, W, n, m) = self.get_variables(data)
        # Using sigmoid as in the original exercise, but here using TF library function.
        h = tf.nn.sigmoid(tf.matmul(X,W)) # As in the original exercise: h = sigmoid(X * theta);
        # Cost function with plain formula as we did in the original exercise.
        # Original formula: J = (1 / m) * ( (((-y)') * (log(h))) - (((1 - y)') * (log(1-h))));
        J_plain_cost = tf.reduce_sum((1/m) *
                                     (tf.subtract(
                                        tf.multiply(-y, tf.log(h)),
                                         tf.multiply(1 - y, tf.log(1 - h)))) )
        # NOTE: GradientDescentOptimizer does not converge properly, tried various alpha values.
        #step = tf.train.GradientDescentOptimizer(alpha).minimize(J_plain_cost)
        step = tf.train.AdamOptimizer(alpha).minimize(J_plain_cost)
        init = tf.global_variables_initializer()
        # For recording cost history.
        J_history = np.empty(shape=[1],dtype=float)
        # Using Python 'with' statement (context manager) this time - no need to close session explicitely later.
        with tf.Session() as session:
            session.run(init)
            for i in range(iterations):
                session.run(step, feed_dict={X: X_train_bias, y: y_train})
                J_history = np.append(J_history,session.run(J_plain_cost,feed_dict={X:X_train_bias,y:y_train}))
            if (self.plotting_enabled):
                self.plot_j_history(J_history, iterations)

            J_value = session.run(J_plain_cost,feed_dict={X:X_train_bias,y:y_train})
            self.logger.debug("Comparing cost with the original cost of the Coursera exercise after {0:d} iterations:"
                              .format(iterations))
            original_cost =  0.203498
            delta = J_value - original_cost
            delta_percentage = (100*(J_value - original_cost)/original_cost)
            self.logger.debug("J_value: {0:.6f}, original cost found by fminunc: {1:.6f}, delta: {2:.6f} ({3:.4f}%)".format(
                J_value, original_cost, delta, delta_percentage))
            final_weights = session.run(W)
            original_weights = [-25.161272, 0.206233, 0.201470] # Original weights (theta) in ex2.m
            self.logger.info("Final trained weights: {0:.4f} (original: {1:.4f}), {2:.4f} (original: {3:.4f}), {4:.4f} (original: {5:.4f})"
                .format(final_weights[0][0], original_weights[0], final_weights[1][0], original_weights[1], final_weights[2][0], original_weights[2]))

            # Make the prediction as in the original Octave exercise using values 45 and 85 (first element is bias = 1).
            X_test = np.asarray([[45.0, 85.0]])
            X_test_bias = np.c_[np.ones(X_test.shape[0]),X_test]
            y_predicted = session.run(h, feed_dict={X: X_test_bias})
            original_result_raw = np.asanyarray([0.776289])
            original_result = original_result_raw.reshape((original_result_raw.shape[0], 1))
            delta = y_predicted - original_result
            delta_percentage = (100*(y_predicted - original_result)/original_result)
            self.logger.info("Comparing to original ex2a predictions using values: exam1 = {0} and exam2 = {1}"
                .format(X_test[0][0], X_test[0][1]))
            self.logger.info("Our prediction: {0:.5f} (original: {1:.5f}), delta: {2:.5f} ({3:.5f}%)".format(
                y_predicted[0][0], original_result[0][0], delta[0][0], delta_percentage[0][0]))

            if (self.plotting_enabled):
                self.plot_data(data, weights=[final_weights[0][0], final_weights[1][0], final_weights[2][0]])

        self.logger.debug("EXIT run_logistic_regression_plain_training_cost")
        return 0  # Everything ok.


    def initialize(self, datafile):
        self.logger.debug("ENTER initialize")
        iterations = int(self.config['DEFAULT']['iterations'])
        alpha = float(self.config['DEFAULT']['alpha'])  # Learning rate.
        self.logger.debug("Iterations: {0}".format(iterations))
        self.logger.debug("Alpha: {0}".format(alpha))
        data = self.read_csv_file(datafile)
        (exam1, exam2, admittance) = data
        dataCount = len(exam1)
        self.logger.debug("Read items: {0}".format(dataCount))
        ml_config = (iterations, alpha)
        return (data, ml_config)


    def run(self, datafile):
        self.logger.debug("ENTER run")
        ret = 0
        self.logger.debug("data file: {0}".format(datafile))
        (data, ml_config) = self.initialize(datafile)
        if (self.plotting_enabled):
            self.plot_data(data)
        ret = self.run_logistic_regression_plain_initial_cost(data, ml_config)
        ret = self.run_logistic_regression_sigmoid_cross_entropy_initial_cost(data, ml_config)
        ret = self.run_logistic_regression_plain_training_cost(data, ml_config)
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
    model = UniversityAdmissionLogisticRegression(config_file, plotting_enabled)
    return_code = model.run(data_file)
    sys.exit(return_code)
