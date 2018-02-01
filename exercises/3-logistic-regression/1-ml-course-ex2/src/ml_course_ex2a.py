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

    def plot_data(self, data):
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
        ax.plot(exam1_passed_v, exam2_passed_v, "bx")
        ax.plot(exam1_not_passed_v, exam2_not_passed_v, "ro", fillstyle='none')
        mplot.show()
        self.logger.debug("EXIT plot_data")
        return


    def run_logistic_regression(self, data):
        """
        Logistic regression ex2 exercise.
        """
        self.logger.debug("ENTER run_logistic_regression")

        self.logger.debug("EXIT run_logistic_regression")
        return 0  # Everything ok.


    def run(self, datafile):
        self.logger.debug("ENTER run")
        ret = 0
        self.logger.debug("data file: {0}".format(datafile))
        data = self.read_csv_file(datafile)
        (exam1, exam2, admittance) = data
        dataCount = len(exam1)
        self.logger.debug("Read items: {0}".format(dataCount))
        if (self.plotting_enabled):
            self.plot_data(data)
        ret = self.run_logistic_regression(data)
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
