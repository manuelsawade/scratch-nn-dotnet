# Simple neural network implementation in c# 
This repository contains a from-scratch implementation of a neural network based on the book 'Neural Networks and Deep Learning'
from Michael Nielson. The source code can be found here: (https://github.com/mnielsen/neural-networks-and-deep-learning). 

## Application
This project is the executable program that loads the data and runs the training for the neural network implementations. Before
the data is passed to the network, its features are scaled around 0 using the standard deviation of 1 and the lables are one-hot-encoded.

## Neural Networks
### Simple Neural Network
The simple neural network implemenation has fixed initializers, activation- and cost-functions. It uses a random bias initializer,
xavier weight initialization, the quadratic cost function and sigmoid activation.  

### Neural Network
This implementation follows a more generic approach and can be used with various initializers, activation- and cost-functions, as long as 
the requirements for the interfaces are met. 

### Accelerated Neural Network
Since the training with the complete training data takes a lot of time, this implementation uses parallel processing and the tensor primitives 
library to boost up the training process.
