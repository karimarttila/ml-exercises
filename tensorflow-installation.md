# TensorFlow Installation Instructions

## Installation Instructions

See: https://www.tensorflow.org/install/install_linux

Use these instructions to install tensorflow in a virtual environment.

### One-time Tasks

These instructions need to be applied only once. You can also use the script create-venv.sh to create the Python Virtual environment (or manually use the steps described below).

Create the virtual environment called tensorflow (once):

```bash
virtualenv --system-site-packages -p python3 venv
```

...or where ever you want to install it; I'm going to install virtualenv in the root of this repo but you don't see it there because it is ignored in my .gitignore file. The same virtualenv will be used in all my TensorFlow exercises.

Activate virtual environment:

```bash
source venv/bin/activate
```

(you should now see (venv) in command prompt)

Ensure pip ≥8.1 is installed (once):

```bash
easy_install -U pip
```

```bash
Install tensorflow:
pip3 install --upgrade tensorflow
```
```bash
echo "Install keras:"
pip3 install --upgrade keras
```

```bash
Install matplotlib (usefull in plotting data):
pip3 install --upgrade matplotlib
```

```bash
Install numpy:
pip3 install --upgrade numpy
```

```bash
Install scipy:
pip3 install --upgrade scipy
```

```bash
Install sklearn:
pip3 install --upgrade sklearn
```

NOTE: sklearn not necessary for exercises, used it a bit to study numpy and scipy.

Test installation ('=>...' not including to the command):

```python
python3
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

=> Should print: 

```bash
b'Hello, TensorFlow!'
```

... or you can use the test script I provided: (assuming you are in root directory of this repo):

```python
python3 exercises/1-tensorflow-installation/src/validate-tensorflow-installation.py
```

If you saw warning like

```
tensorflow/core/platform/cpu_feature_guard.cc:137] 
Your CPU supports instructions that this TensorFlow 
binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
``` 
... then:
 
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
```

I.e. the default is 0 (all logs), 2 filters out warnings.
Another solution is to build tensorflow from sources.

Deactivate virtual environment once you are done:

```bash
deactivate
```

### Install Python tk Package to Linux

I realized that I can plot the matplotlib graphics in PyCharm but not in command line. For plotting graphis in command line you have to install python3-tk:

```bash
sudo apt-get install python3-tk
```

### Start Using TensorFlow in a Virtual Environment

Activate virtual environment:

```bash
source venv/bin/activate
```

(you should now see (tensorflow) in command prompt)

Deactivate virtual environment once you are done:

```bash
deactivate
```

### Next Steps

Continue configuring PyCharm if you use that Python IDE: see file pycharm-instructions.md.
