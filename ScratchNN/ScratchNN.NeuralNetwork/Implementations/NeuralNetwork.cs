using ScratchNN.NeuralNetwork.Activations;
using ScratchNN.NeuralNetwork.CostFunctions;
using ScratchNN.NeuralNetwork.Extensions;
using System.Diagnostics;
using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork.Implementations;

public class NeuralNetwork : NeuralNetworkBase
{
    private readonly Random _random = new();
    private readonly ICostFunction _costFunction;
    private readonly IActivationFunction _activationFunction;

    public override int Seed { get; init; }
    public override int[] Layers { get; init; }
    public override float[][] Biases { get; init; }
    public override float[][][] Weights { get; init; }

    public NeuralNetwork(
        int[] layers,
        ICostFunction costFunction,
        IActivationFunction activationFunction,
        int? seed = null)
    {
        _costFunction = costFunction;
        _activationFunction = activationFunction;

        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = InitBiases(layers, _random);
        Weights = InitWeights(layers, _random);
    }

    public NeuralNetwork(
        int[] layers,
        float[][] biases,
        float[][][] weights,
        ICostFunction costFunction,
        IActivationFunction activationFunction,
        int? seed = null)
    {
        _costFunction = costFunction;
        _activationFunction = activationFunction;

        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = biases;
        Weights = weights;
    }

    public float[] Predict(float[] inputData) => FeedForward(inputData).Outputs[^1];


    public (float[][] Outputs, float[][] WeightedSums) FeedForward(float[] inputData)
    {
        var outputs = Layers.New<float>();
        outputs[0] = inputData;

        var weightedSums = Layers.New<float>();

        for (var iLayer = 1; iLayer < Layers.Length; iLayer++)
        {
            for (var iNeuron = 0; iNeuron < Layers[iLayer]; iNeuron++)
            {
                var inputs = outputs[iLayer - 1];
                var weights = Weights[iLayer][iNeuron];
                var bias = Biases[iLayer][iNeuron];
                var weightedInput = 0f;

                for (var iInput = 0; iInput < inputs.Length; iInput++)
                {
                    weightedInput += inputs[iInput] * weights[iInput];
                }

                var weightedSum = weightedInput + bias;

                outputs[iLayer][iNeuron] = _activationFunction.Activation(weightedSum);
                weightedSums[iLayer][iNeuron] = weightedSum;
            }
        }

        return (outputs, weightedSums);
    }

    public void Fit(
        LabeledData[] trainingData,
        int epochs,
        int batchSize,
        float learningRate,
        float regularization)
    {
        var validationSetLength = (int)(trainingData.Length * 0.1);
        var validationData = trainingData
                .Shuffle(_random)
                .Take(validationSetLength)
                .ToArray();

        trainingData = trainingData.Skip(validationSetLength).ToArray();

        foreach (var epoch in Enumerable.Range(0, epochs))
        {
            var miniBatches = trainingData
                .Shuffle(_random)
                .Chunk(batchSize)
                .ToArray();

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (var iBatch = 0; iBatch < miniBatches.Length; iBatch++)
            {
                Console.Write($"Epoch {epoch,2} | Fit Batches: {iBatch}/{miniBatches.Length}");
                UpdateParameters(miniBatches[iBatch], learningRate, regularization);

                ConsoleExtensions.ClearCurrentLine();
            }

            Evaluate(validationData, regularization);

            stopwatch.Stop();
            var (accuracy, cost) = Evaluate(validationData, regularization);

            Console.WriteLine($"Accuracy: {accuracy,-4} | Cost: {cost,-6} | Elapsed: {stopwatch.Elapsed}");
        }
    }

    public void UpdateParameters(LabeledData[] trainingBatch, float learningRate, float regularization)
    {
        var costSumBias = Biases.Shape().New<float>();
        var costSumWeights = Weights.Shape().New<float>();

        foreach (var (InputData, Expected) in trainingBatch)
        {
            var (costsBias, costsWeights) = Backpropagation(InputData, Expected);

            costSumBias = costSumBias.Add(costsBias);
            costSumWeights = costSumWeights.Add(costsWeights);
        }

        IterateNetwork(
            biasAction: (iLayer, iNeuron, bias) =>
            {
                Biases[iLayer][iNeuron] =
                    bias - learningRate * costSumBias[iLayer][iNeuron] / trainingBatch.Length;
            },
            weightAction: (iLayer, iNeuron, iWeight, weight) =>
            {
                Weights[iLayer][iNeuron][iWeight] =
                    (1 - learningRate * (regularization / trainingBatch.Length)) *
                    weight - learningRate * costSumWeights[iLayer][iNeuron][iWeight] / trainingBatch.Length;
            });
    }

    public (float[][], float[][][]) Backpropagation(float[] inputData, float[] expected)
    {
        var costsBias = Biases.Shape().New<float>();
        var costsWeights = Weights.Shape().New<float>();

        var (outputs, weightedSum) = FeedForward(inputData);

        var costs = _costFunction.Cost(outputs[^1], expected, _activationFunction.Gradient(weightedSum[^1]));

        costsBias[^1] = costs;
        costsWeights[^1] = costs.Multiply(weightedSum[^2].Transpose());

        for (var iLayer = Layers.Length - 2; iLayer > 0; iLayer--)
        {
            costs = Weights[iLayer + 1]
                .Transpose()
                .Multiply(costs)
                .Multiply(_activationFunction.Gradient(weightedSum[iLayer]));

            costsBias[iLayer] = costs;
            costsWeights[iLayer] = costs.Multiply(outputs[iLayer - 1].Transpose());
        };

        return (costsBias, costsWeights);
    }

    public (float, float) Evaluate(LabeledData[] labeledTestData, float regularization)
    {
        var correctPredictions = 0;
        var costs = new float[labeledTestData.Length];

        Parallel.For(0, labeledTestData.Length, new() { MaxDegreeOfParallelism = 1 }, (iData) =>
        {
            var (inputData, expected) = labeledTestData[iData];
            var output = Predict(inputData);

            if (TensorPrimitives.IndexOfMax(output) == TensorPrimitives.IndexOfMax(expected)) //output[TensorPrimitives.IndexOfMax(expected)] == TensorPrimitives.Max(output))
            {
                Interlocked.Increment(ref correctPredictions);
            }

            costs[iData] = _costFunction.Computation(output, expected);
        });

        var accuracy = (float)Math.Round(correctPredictions / (double)labeledTestData.Length, 2);

        var completeCost = 0.5f * (regularization / labeledTestData.Length) * Weights[1..].Norm() + costs.Sum();

        return (accuracy, completeCost);
    }
}
