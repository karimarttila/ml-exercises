"""
Microchip test logistic regression analysis using TensorFlow."
Usage: python3 ml_course_ex2b.py <data-file> <config-file> <plot data: true/false>"

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


class MicrochipLogisticRegression:
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
        test1 = []
        test2 = []
        accepted = []
        with open(datafile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                test1.append(float(row[0]))
                test2.append(float(row[1]))
                accepted.append(int(row[2]))
        self.logger.debug("EXIT read_csv_file")
        return (test1, test2, accepted)


    def plot_data(self, data, weights=None):
        self.logger.debug("ENTER plot_data")
        (test1, test2, accepted) = data
        test1_v = np.asarray(test1)
        test2_v = np.asarray(test2)
        accepted_v = np.asarray(accepted)
        accepted_passed_v = (accepted_v == 1)
        accepted_not_passed_v = (accepted_v == 0)
        test1_passed_v = np.extract(accepted_passed_v, test1_v)
        test2_passed_v = np.extract(accepted_passed_v, test2_v)
        test1_not_passed_v = np.extract(accepted_not_passed_v, test1_v)
        test2_not_passed_v = np.extract(accepted_not_passed_v, test2_v)
        fig = mplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Test1 score')
        ax.set_ylabel('Test2 score')
        major_ticks = np.arange(0, 101, 10)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.plot(test1_passed_v, test2_passed_v, "bx", label= 'Accepted')
        ax.plot(test1_not_passed_v, test2_not_passed_v, "ro", fillstyle='none', label= 'Not accepted')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        mplot.show()
        self.logger.debug("EXIT plot_data")
        return


    def run_logistic_regression(self, data, ml_config):
        """
        Logistic regression ex2b exercise.
        NOTE: In the original ML Coursera course the data was NOT normalized,
        so we don't normalize it in this exercise either.
        """
        self.logger.debug("ENTER run_logistic_regression")


    def initialize(self, datafile):
        self.logger.debug("ENTER initialize")
        iterations = int(self.config['DEFAULT']['iterations'])
        alpha = float(self.config['DEFAULT']['alpha'])  # Learning rate.
        self.logger.debug("Iterations: {0}".format(iterations))
        self.logger.debug("Alpha: {0}".format(alpha))
        data = self.read_csv_file(datafile)
        (test1, test2, accepted) = data
        dataCount = len(test1)
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
        ret = self.run_logistic_regression(data, ml_config)
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
    model = MicrochipLogisticRegression(config_file, plotting_enabled)
    return_code = model.run(data_file)
    sys.exit(return_code)
