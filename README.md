# A Deep Neural Network for Flower Classification
This was my submission for the Udacity [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089). The project required students to build a deep neural network to classify pictures of flowers.

The project involved importing a densenet161/vgg16/other architecture, then stripping away the classifier module and implementing a naive classifier to be trained on recognising 102 categories of flowers. The code was written to run from the command line by passing arguments specifying architecture, training epochs & other hyperparameters.

A [Jupyter Notebook is provided](https://github.com/andrefmsmith/udacity_flowerclassifier/blob/master/Image%20Classifier%20Project_notebook_cleaned.ipynb) which served as the development structure for the final project. The latter uses the python argparse library to run from the command line.

There are some differences in implementation between the notebook and the final command line project. I tried to make the latter more robust and reliant, whenever possible, on object-oriented programming rather than hard-coded values, whereas the former was really a testbed for trying out ideas, architectures, hyperparameters and so on.

[Train.py](https://github.com/andrefmsmith/udacity_flowerclassifier/blob/master/train.py) is a script that calls the necessary functions for structuring data, importing, instantiating and training a deep network and saves a checkpoint for later use. 

[Predict.py](https://github.com/andrefmsmith/udacity_flowerclassifier/blob/master/predict.py) picks up this checkpoint, reinstates the network and, given a path to an image, calls functions to process it, run it through the trained classifier, and retrieve the top k class assignments and their probability as judged by the network. Both scripts rely on [model](https://github.com/andrefmsmith/udacity_flowerclassifier/blob/master/model_functions.py) and [utility](https://github.com/andrefmsmith/udacity_flowerclassifier/blob/master/utility_functions.py) functions I hereby link to.



I hope this is useful to you in your journey into AI and Deep Neural Networks and I can greatly endorse the Nanodegree on Udacity as a way to get started with AI programming. I would also highly recommend Andrew Ng's excellent courses, namely the Coursera (despite being Matlab/Octave-based ;)) [Machine Learning](https://www.coursera.org/learn/machine-learning) and his [Deeplearning.ai specialisation on Deep Learning](https://www.deeplearning.ai/).
