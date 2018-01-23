#!/bin/bash

echo "********* Create Python3 virtual environment..."
virtualenv --system-site-packages -p python3 venv

echo "********* Activate the virtual environment..."
source venv/bin/activate

echo "********* Install pip..."
easy_install -U pip

echo "********* Install tensorflow..."
pip3 install --upgrade tensorflow

echo "********* Install keras..."
pip3 install --upgrade keras

echo "********* Install matplotlib..."
pip3 install --upgrade matplotlib

echo "********* Install numpy..."
pip3 install --upgrade numpy

echo "********* Install scipy..."
pip3 install --upgrade scipy

echo "********* Install sklearn..."
pip3 install --upgrade sklearn

echo "********* Finally leave virtual environment..."
deactivate
