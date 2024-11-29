# Simple neural network implementation in c# 
This repository contains a from-scratch implementation of a neural network based on the book 'Neural Networks and Deep Learning'
from Michael Nielson. The source code can be found here: (https://github.com/mnielsen/neural-networks-and-deep-learning). 

The Solution contains 3 projects and is structured as followed: 
- ScratchNN.App
- ScratchNN.NeuralNetwork
- ScratchNN.NeuralNetwork.Tests

## [Application](https://github.com/manuelsawade/scratch-nn-dotnet/tree/main/ScratchNN/ScratchNN.App)
This project is the executable program that loads the data and runs the training for the neural network implementations. Before
the data is passed to the network, its features are scaled around 0 using the standard deviation of 1 and the lables are one-hot-encoded.

## [Neural Networks](https://github.com/manuelsawade/scratch-nn-dotnet/tree/main/ScratchNN/ScratchNN.NeuralNetwork)
### [Simple Neural Network](https://github.com/manuelsawade/scratch-nn-dotnet/blob/main/ScratchNN/ScratchNN.NeuralNetwork/Implementations/SimpleNeuralNetwork.cs)
The simple neural network implemenation has fixed initializers, activation- and cost-functions. It uses a random bias initializer,
xavier weight initialization, the quadratic cost function and sigmoid activation.  

### [Neural Network](https://github.com/manuelsawade/scratch-nn-dotnet/blob/main/ScratchNN/ScratchNN.NeuralNetwork/Implementations/NeuralNetwork.cs)
This implementation follows a more generic approach and can be used with various initializers, activation- and cost-functions, as long as 
the requirements for the interfaces are met. 

### [Accelerated Neural Network](https://github.com/manuelsawade/scratch-nn-dotnet/blob/main/ScratchNN/ScratchNN.NeuralNetwork/Implementations/AcceleratedNeuralNetwork.cs)
Since the training with the complete training data takes a lot of time, this implementation uses parallel processing and the 
[tensor primitives](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives?view=net-9.0-pp)
library to boost up the training process.

## [Tests](https://github.com/manuelsawade/scratch-nn-dotnet/tree/main/ScratchNN/ScratchNN.NeuralNetwork.Tests)
This project contains integration tests securing the functionality of backpropagation and updating the parameters for all neural network implementations.
