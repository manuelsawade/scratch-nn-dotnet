using Microsoft.Extensions.Configuration;
using ScratchNN.App;
using ScratchNN.NeuralNetwork;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .Build();

var (trainingData, testData) = DataPreparation.Prepare(config);

var neuralnetwork = new NeuralNetwork([784, 30, 10]);

neuralnetwork.Fit(trainingData[..10_000], 10, 8, 0.05f, 0.01f);
neuralnetwork.Evaluate(testData);

//var neuralnetwork = new SimpleNeuralNetwork([784, 30, 10]);

//neuralnetwork.Fit(trainingData[..5_000], 10, 8);
//neuralnetwork.Evaluate(testData);
