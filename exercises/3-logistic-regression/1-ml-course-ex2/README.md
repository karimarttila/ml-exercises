# Logistic Regression - Coursera ML Exercise 2 Using TensorFlow

## Introduction

This is the ex2 in the [Machine Learning](https://www.coursera.org/learn/machine-learning) course provided in [Coursera](https://www.coursera.org). I did in winter 2016/2017. 

I have divided the actual exercise 2 into smaller parts (a,b...) in the order the parts of the original exercise was described in the PDF instruction.

## Exercise 2a

### Introduction

In the original exercise we implemented a simple logistic regression model to predict whether an applicant gets admitted into a university based on the results of two exams.

As in the previous ML Coursera course exercise I plot the data in the beginning of the program to visualize the students who passed admittance (blue crosses) vs. those who didn't pass (red circles) regarding how they did in exam1 and exam2. This was once again an interesting exercise using numpy to create the boolean arrays regarding the admittance value (0 or 1) and then to create two sets of arrays exam1 and exam2 using the boolean arrays as filters. 

![ex2 plot](images/ex2a-university-admittance-plot-python.png "ex2 plot")

You can create the graphics running the program as:

```bash
./run-ex2a.sh data/ex2a-university-exam-results.csv ml_course_ex2a.ini true
```

So, as one can see you had to make pretty well in both exams to get admitted to the university. If you got excellent scores in one of the exam but did poorly in the other one your chances for getting admitted is not good.


### Implementation

I did the exercise pretty much the same way as the original exercise, except I implemented the logistic regression cost function twice: the first implementation is exactly the same plain cost formula as in the original exercise, the second implementation uses TensorFlow [sigmoid_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits). The reason for this was that I wanted to compare the two ways to implement logistic regression. 


### Analysis

#### Plain Cost Function

The plain cost function gave the same results as in the original exercise:

```bash
2018-02-08 18:59:22,269 - TF - DEBUG - Comparing cost with the original cost of the Coursera exercise:
2018-02-08 18:59:22,270 - TF - DEBUG - J_value: 0.693147, original cost: 0.693150, delta: -0.000003 (-0.0004%)
```

