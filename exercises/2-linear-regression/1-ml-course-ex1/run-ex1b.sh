#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: ./run-ex1b.sh <data-file> <config-file> <plot data: true/false>"
    echo "Examples:"
    echo "./run-ex1b.sh data/ex1b-real-estate-transactions.csv ml_course_ex1b.ini false"
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

if [ ! -f $CONFIG_FILE ]
then
    echo "File $CONFIG_FILE does not exists, exiting..."
    exit 3
fi


PYTHONPATH=$LOGGER_DIR python3 src/ml_course_ex1b.py $DATA_FILE $CONFIG_FILE $PLOT_DATA
