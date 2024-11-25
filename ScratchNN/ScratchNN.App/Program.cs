using Microsoft.Extensions.Configuration;
using ScratchNN.App;
using ScratchNN.NeuralNetwork;
using ScratchNN.NeuralNetwork.Activations;
using ScratchNN.NeuralNetwork.CostFunctions;
using System.Diagnostics;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .Build();

var (trainingData, testData) = DataPreparation.Prepare(config);

//var fastnetwork = new CPUAcceleratedNeuralNetwork(
//    [784, 100, 10],
//    new CrossEntropyCost(),
//    new SigmoidActivation(),
//    4321);

//fastnetwork.Fit(trainingData[..10_000], 50, 10, 0.0001f, 0.1f);
//var (accuracy, cost) = fastnetwork.Evaluate(testData, 0.1f);
//Console.WriteLine($"Test | Accuracy: {accuracy,-4} | Cost: {cost,-6}");

//var neuralnetwork = new NeuralNetwork(
//    [784, 30, 10],
//    new CrossEntropyCost(),
//    new SigmoidActivation());

//neuralnetwork.Fit(trainingData[..1_000], 10, 10, 10f, 20f);
//neuralnetwork.Evaluate(testData, 5f);

var simpleneuralnetwork = new SimpleNeuralNetwork([784, 100, 10]);

simpleneuralnetwork.Fit(trainingData[..10_000], 50, 10, 0.0001f);
simpleneuralnetwork.Evaluate(testData);
