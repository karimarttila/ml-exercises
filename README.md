# Machine Learning Exercises

## Introduction

I gather here various Machine learning exercises I do while studying Machine learning.

This Github project is basically the step two I describe in my Machine learning study path Medium blog article, [Studying Machine Learning — First Impressions](https://medium.com/@kari.marttila) . The first step (theory) was attending the excellent [Machine Learning](https://www.coursera.org/learn/machine-learning) course provided in [Coursera](https://www.coursera.org). I really can recommend the course since prof. Andrew Ng is an excellent teacher, he makes the rather difficult theory quite understandable using various examples and there are a lot of good exercises that makes the theory more understandable when you build the ML algorithms yourself using simple low level primitives (like matrix calculation). After the course I believe you understand better what is happening under the hood when using more higher level ML frameworks and libraries. 

## Which Tools Am I Using?

I use both [Octave](https://www.gnu.org/software/octave/) and [TensorFlow](https://www.tensorflow.org/) to implement various Machine learning models I use in my ML exercises. 

The versions of the tools I have used when making these exercises are:

- Octave: 3.8.1
- Python: 3.4.3
- TensorFlow: 1.4.1

I have been running Ubuntu 14 when I did these exercises.

## Two kinds of exercises

There are two kinds of exercises in this repository. 

1. I try to implement the most important exercises given in above mentioned [Machine Learning](https://www.coursera.org/learn/machine-learning) course in TensorFlow. I have implemented the solutions with the help of instructions provided by the course using Octave. I don't give the Octave solutions here, of course, because the Coursera Honor Code denies that. But I thought it might be an interesting exercise for me to implement the same solutions with industry-level ML library because I now know pretty well how the same solution was implemented using low level calculus (the solutions are pretty different - in ML course we implemented algorithms e.g. using matrix calculation, in my TensorFlow exercises I use library provided functions - this should not break the Honor Code).

2. Real world problems using open data. I try to find some real world problems for which I find open data. Using real world problems makes the exercises more interesting and hopefully I can help some Finnish Agencies analyzing their data for the gratitude that they have opened the data for public. For these exercises I first try to create a simple model in Octave to validate my ideas. Then I try to create a more professional model using Python/TensorFlow. The TensorFlow model can be integrated to a real application (using e.g. Python, Java or Clojure). If I have time I try to create a short demonstration of a backend service that uses the ML model and maybe a simple frontend-application fing the backend service.


## Github Project

This project is hosted in Github in url [ml-exercises](https://github.com/karimarttila/ml-exercises).

## Directories

- data: Various raw and processed material from various open data sources. Processed data will be used as training and test sets. 
- exercises: The exercises. I have created subdirectories for those ML algorithms I'm experimenting. There in each ML algorithm directory there can be 1-N different exercises. Some of the exercises have just TensorFlow solutions. Some both TensorFlow and Octave solutions (in different subdirectories).

## Disclaimer

These exercises have been done just for my own learning purposes. I have tried to make the exercises so that anyone can run the exercises using Octave / Python in their own Linux machines (only bash scripts to run the exercises are provided) but I don't promise they work in your workstation without some tweaking. So, do not send me any email like "exercise X does not run in my computer". :-)

Also, it is very possible that the models in these exercises have various defects or errors. Machine learning is hard and I am studying Machine learning while making these exercises - it is quite understandable that you make errors when you are learning something. 

## My Links

- [Kari Marttila in LinkedIn](https://www.linkedin.com/in/karimarttila)
- [Kari Marttila Medium Blog](https://medium.com/@kari.marttila)

