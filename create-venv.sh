#!/bin/bash

echo "Create Python3 virtual environment..."
virtualenv --system-site-packages -p python3 venv

echo "Activate the virtual environment..."
source venv/bin/activate

echo "Install pip..."
easy_install -U pip

echo "Install tensorflow..."
pip3 install --upgrade tensorflow

echo "Install matplotlib..."
pip3 install --upgrade matplotlib

echo "Finally leave virtual environment..."
deactivate
