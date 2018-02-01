# Linear Regression - Coursera ML Exercise 1 Using TensorFlow

## Introduction

This is the ex1a in the [Machine Learning](https://www.coursera.org/learn/machine-learning) course provided in [Coursera](https://www.coursera.org). I did in winter 2016/2017. 

I have divided the actual exercise 1 into smaller parts (a,b...) in the order the parts of the original exercise was described in the PDF instruction.

## Exercise 1a

### Introduction

In the original exercise we implemented a simple linear regression model of one variable to predict profits for a food truck related to the population of the cities (the data should be interpreted x1000). 

The program has capability to plot the data using matplotlib.pyplot library. I took a screenshot of this plot which shows the relation between the population of the city and the profits using the data provided.

![Octave plot of ex1](images/ex1a-population-profit-plot-python.png "Octave plot of ex1")

You can create the graphics running the program as:

```bash
./run-ex1a.sh data/ex1a-profit-population.csv ml_course_ex1a.ini true
```

In the data directory you can find the data file used in this exercise. The 0th column provides X = population, the 1st column provides y = profit.

This is linear regression of one variable (X) related to y, i.e. y = f(X). 

### Implementation

I modularize every exercise to a Python class which makes IMHO the code a bit more readable. At the end of the file is the main entry point to the file. It basically just reads the command line arguments, instantiates the class and calls the run method. 

The run method first plots the original data and then starts the linear regression calculation using TensorFlow library (with some help of NumPy library). The function is rather long, but this is not meant to be production software, just my private exercise. 

The implementation should be rather self-descriptive if you have learned the basics of linear regression and some basics of TensorFlow and NumPy libraries. Once the basic stuff (bias, placeholders, variables...) are done the actual heavy lifting is done using TensorFlow's [GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer) function (in the Coursera course we had to implement the algorithm using primitive matrix calculation).

### Analysis

The analysis part of the exercise was pretty interesting since I had done the same exercise previously in the ML course using Octave. 

The following picture shows the convergance of the cost function (J in code, J seemed to be the name prof. Ng used for cost functions, I mostly follow the same terminology in my Python code, i.e. J is cost function, n is the number of features and m is the number of samples).

![Convergance of Python exercise](images/ex1a-convergance-python.png "Convergance of Python exercise")

I added some logging to compare this exercises values to the original ex1a exercise I did earlier using Octave:

```bash
2018-01-23 10:54:43,383 - TF - INFO - Comparing to original ex1 predictions using populations 3.5, 7.0 and 20.0
2018-01-23 10:54:43,383 - TF - INFO - Population: 3.5, profits: our prediction: 0.490537 (original: 0.451977), delta: 0.04 (8.53%)
2018-01-23 10:54:43,383 - TF - INFO - Population: 7.0, profits: our prediction: 4.551895 (original: 4.534245), delta: 0.02 (0.39%)
2018-01-23 10:54:43,383 - TF - INFO - Population: 20.0, profits: our prediction: 19.636938 (original: 19.696956), delta: -0.06 (-0.30%)
2018-01-23 10:54:43,384 - TF - INFO - Final trained weights: -3.5708 (original: -3.6303), 1.1604 (original: 1.1664)
2018-01-23 10:54:43,385 - TF - INFO - Convergance: J[1]: 6.7397 (original: 6.7372), J[1500]: 4.4866 (original: 4.4834)
```

So, there is some delta between the TensorFlow version predictions and original ex1 predictions done using Octave. First I was wondering whether there was some issue with the convergance, but when I compared the convergance values (the first and last cost function values between the TensorFlow version and Octave version) I realized that they are practically the same. Maybe I analyze this later on but let's now continue to the next exercise.

The original ex1 exercise plots the regression line to the diagram. I experimented with numpy and matplotlib to do this. I created the regression line (green) to the diagram and also calculated the predicted profit for each population value and plotted the predicted profits using red crosses (original values are in blue crosses). So, the red crosses basically describe how the population values are projected to the green regression line.

![Regression line of Python exercise](images/ex1a-regression-line-python.png "Regression line of Python exercise")


### Reflections Regarding Python Programming

It's been a while since I previously did some serious Python programming. I have been programming production software mostly in Java, and the last year also using Clojure (see more regarding my Clojure experiences [here](https://medium.com/tieto-developers/clojure-impressions-round-two-f989c0945f4b) ). 

Python as a language is pretty easy and as an interpreted language experimenting ML model and plotting data is rather easy. Python also provides rather good REPL (but nothing compared to Clojure REPL, see examples of Cursive REPL [here](https://cursive-ide.com/userguide/repl.html)).

If you want to experiment with the code using Python repl you can start it so that you add the utils directory to the python path:

```bash
pwd
<your-cloned-directory>/exercises/2-linear-regression/1-ml-course-ex1
PYTHONPATH=../../../utils python3
```

Once you are in the Python REPL you can interact with the code, e.g:

```bash
>>> import src.ml_course_ex1a as ex1a
>>> model = ex1a.ProfitPopulationLinearRegression("ml_course_ex1a.ini", True)
>>> (populations,profits) = model.readCsvFile("data/ex1a-profit-population.csv")
2018-01-23 21:32:06,418 - TF - DEBUG - ENTER readCsvFile
2018-01-23 21:32:06,419 - TF - DEBUG - EXIT readCsvFile
>>> len(populations)
97
>>> import numpy as np
>>> populations_array = np.asarray(populations)
>>> populations_array
array([ 6.1101,  5.5277,  8.5186,  7.0032,  5.8598,  ... 
>>> X_train = np.asarray(populations)
>>> X_train_bias = model.appendBias(X_train)
2018-01-24 20:38:26,672 - TF - DEBUG - ENTER appendBias
2018-01-24 20:38:26,673 - TF - DEBUG - EXIT appendBias
>>> X_train.shape
(97,)
>>> X_train_bias.shape
(97, 2)
>>> X_train_bias
array([[ 1.    ,  6.1101],
       [ 1.    ,  5.5277],
       [ 1.    ,  8.5186],
...
>>> dir()
['X_train', 'X_train_bias', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__',
...
# dir() is useful, you can print the variables you have in your REPL.
```

So, you can create Python code interactively using the Python REPL and once you are happy, you copy-paste the code to your source file. Using REPL like this is quite powerful way to create code since you can create the code in small blocks.

If you like working with the Python REPL I really recommend you to learn [Clojure](https://clojure.org/). Once you master Clojure you realize how powerful Clojure REPL is since it has all the power of Lisp and its S-expressions. If you think that using Python REPL is productive you realize that Clojure REPL is 10X as productive. 

Using a good IDE makes programming easier, of course. If you program Python I strongly recommend [PyCharm](https://www.jetbrains.com/pycharm/). See repository root directory for more information about PyCharm.


## Exercise 1b

### Introduction

I actually didn't do the second part of the original Coursera exercise last autumn because it was optional. But I did it now. After doing all exercises using Octave is now a lot easier and the exercise was pretty easy (the formulas basically the same as in exercise 1a (matrix solution - works with any feature set n) or the formulas were given in the exercise instruction pdf and you just had to convert the formula in Octave matrix operations). 

The data set comprises of real estate transactions. Every transaction has three values:

- Area of the real estate.
- Rooms of the real estate.
- Transaction price.

The idea is to create multi variable linear regression model that predicts the price when given an area and number of rooms. 

Let's start again by plotting the data. I have been learning the excellent matplotlib library while doing these exercises and using matplotlib library with Python is really easy.

![Areas and rooms relation to price](images/ex1b-area-rooms-price-3d-plot.png "Areas and rooms relation to price")


### Implementation

The implementation of ex1b is pretty much the same as ex1a - I could have refactored common parts to some library if this were production code; but it is just an exercise so, let's not bother about refactoring too much and move on. 

The run method first plots the original data in 3D and then starts the linear regression calculation using TensorFlow library (with some help of NumPy library) as in previous ex1a.

### Analysis

The analysis part of the exercise was again pretty interesting since I did the original exercise using Octave just before this exercise. 

The following graphics shows the convergance of the cost function:

![Convergance of Python exercise ex1b](images/ex1b-convergance-python.png "Convergance of Python exercise ex1b")

The convergance looks pretty much the same as in the original Octave exercise.

Let's print the results:

```bash
2018-01-25 19:33:48,312 - TF - INFO - Comparing to original ex1b predictions using area 1650.0 and room 3.0
2018-01-25 19:33:48,312 - TF - INFO - Our prediction: 289221.56 (original: 289314.62), delta: -93.06 (-0.03%)
2018-01-25 19:33:48,314 - TF - INFO - Final trained weights: 334302.0938 (original: 334302.0640), 99411.4531 (original: 100087.1160), 3267.0137 (original: 3673.5485)
2018-01-25 19:33:48,314 - TF - INFO - Convergance: J[1]: 64297283584.00 (original: 64300749594.57), J[400]: 2105448192.00 (original: 2108850058.40)
2018-01-25 19:33:48,315 - TF - DEBUG - EXIT run
```

This exercise was interesting also in that sense that I didn't actually submit this ex1b part in the Coursera course (and I didn't submit it now). So, I cannot be 100% sure that the Octave results are correct. But since I got pretty much the same results using Octave (with simple matrix calculation) and using Python / TensorFlow / [GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer) I'm pretty sure that I did both the exercises correctly (or I did both wrong and also the same way wrong). 

It might be interesting to plot the regression plane as I did in ex1a the regression line, but I explored this a bit and decided that it's not now worth the effort. Let's leave this part for some future exercise. Example code how to do that: [Regression plane instructions using matplotlib in Stackoverflow](https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points) (I copy-pasted the code in Python REPL and the code draw a nice regression plane).

## Conclusions

The exercises ex1a and ex1b were pretty interesting to do using Tensorflow. The solutions of feature count n = 1 (in ex1a) and n = 2 (in ex1b) are pretty much the same thanks to Coursera Machine learning instructions how the matrices should be laid out for gradient descent calculation. 

TensorFlow as a library is not that difficult to use. Also the basic matrix operations using numpy were quite self-descriptive and easy to learn. 

I must thank prof. Ng for explaining the material (e.g. the linear regression used in these exercises) in the Coursera course very well. Watching the videos and reading the lecture notes one can grasp the intuition of linear regression and also the more detailed mathematical operations how to make the linear regression model (basically how to find the theta (weights) using gradient descent). Without this understanding creating the linear regression model using TensorFlow library would be rather awkward: you would create the model withour really understanding what is happening under the hood. 

I'll next move to implement the Coursera Machine learning week 3 exercises using Python / TensorFlow. I'll hope that after these exercises I should be rather fluent using NumPy and TensorFlow to create my own models in various real projects.



