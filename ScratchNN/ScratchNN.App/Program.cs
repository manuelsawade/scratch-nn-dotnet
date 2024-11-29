using Microsoft.Extensions.Configuration;
using ScratchNN.App;
using ScratchNN.NeuralNetwork.Activations;
using ScratchNN.NeuralNetwork.CostFunctions;
using ScratchNN.NeuralNetwork.Implementations;
using ScratchNN.NeuralNetwork.Initializers.Biases;
using ScratchNN.NeuralNetwork.Initializers.Weights;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .Build();

var (trainingData, testData) = DataPreparation.Prepare(config);

TrainSimpleNeuralNetwork(trainingData, testData);
TrainNeuralNetwork(trainingData, testData);
TrainAcceleratedNeuralNetwork(trainingData, testData);

void TrainSimpleNeuralNetwork(LabeledData[] trainingData, LabeledData[] testData)
{
    var neuralnetwork = new SimpleNeuralNetwork([784, 100, 10]);

    neuralnetwork.Fit(trainingData[..10_000], 50, 10, 0.05f);
    var accuracy = neuralnetwork.EvaluateAccuracy(testData);
    Console.WriteLine($"Test | Accuracy: {accuracy,-4}");
}

void TrainNeuralNetwork(LabeledData[] trainingData, LabeledData[] testData)
{
    var neuralnetwork = new NeuralNetwork(
        [784, 100, 100, 10],
        new RandomInitializer(),
        new XavierInitializer(),
        new QuadraticCost(),
        new SigmoidActivation());

    neuralnetwork.Fit(trainingData[..10_000], 50, 10, 0.01f, 0.1f);
    var accuracy = neuralnetwork.EvaluateAccuracy(testData);
    Console.WriteLine($"Test | Accuracy: {accuracy,-4}");
}

void TrainAcceleratedNeuralNetwork(LabeledData[] trainingData, LabeledData[] testData)
{
    var neuralnetwork = new AcceleratedNeuralNetwork(
        [784, 100, 10],
        new ConstantInitializer(value: 0.01f),
        new HeInitializer(),
        new CrossEntropyCost(),
        new SigmoidActivation(),
        4321);

    neuralnetwork.Fit(trainingData[..10_000], 50, 10, 0.0001f, 0.1f);
    var accuracy = neuralnetwork.EvaluateAccuracy(testData);
    Console.WriteLine($"Test | Accuracy: {accuracy,-4}");
}


