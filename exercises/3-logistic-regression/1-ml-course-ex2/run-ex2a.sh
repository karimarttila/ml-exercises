#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: ./run-ex1a.sh <data-file> <config-file> <plot data: true/false>"
    echo "Examples:"
    echo "./run-ex2a.sh data/ex2a-university-exam-results.csv ml_course_ex2a.ini false"
    echo "NOTE: Remember to activate the Python3 virtual environment!"
    exit 1
fi

DATA_FILE=$1
CONFIG_FILE=$2
PLOT_DATA=$3
LOGGER_DIR=../../../utils

if [ ! -f $DATA_FILE ]
then
    echo "File $DATA_FILE does not exists, exiting..."
    exit 2
fi

PYTHONPATH=$LOGGER_DIR python3 src/ml_course_ex2a.py $DATA_FILE $CONFIG_FILE $PLOT_DATA
