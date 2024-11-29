using ScratchNN.NeuralNetwork.Activations;
using ScratchNN.NeuralNetwork.CostFunctions;
using ScratchNN.NeuralNetwork.Extensions;
using ScratchNN.NeuralNetwork.Initializers;
using System.Diagnostics;
using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork.Implementations;

public class NeuralNetwork : NeuralNetworkBase
{
    private readonly Random _random = new();
    private readonly ICostFunction _cost;
    private readonly IActivationFunction _activation;

    public override int Seed { get; init; }
    public override int[] Layers { get; init; }
    public override float[][] Biases { get; init; }
    public override float[][][] Weights { get; init; }

    public NeuralNetwork(
        int[] layers,
        IBiasInitializer biasInitializer,
        IWeightInitializer weightInitializer,
        ICostFunction cost,
        IActivationFunction activation,
        int? seed = null)
    {
        (_cost, _activation) = (cost, activation);
        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = InitBiases(layers, _random, biasInitializer);
        Weights = InitWeights(layers, _random, weightInitializer);
    }

    public NeuralNetwork(
        int[] layers,
        float[][] biases,
        float[][][] weights,
        ICostFunction cost,
        IActivationFunction activation,
        int? seed = null)
    {
        (_cost, _activation) = (cost, activation);
        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = biases;
        Weights = weights;
    }

    public override float[] Predict(float[] inputData)
    {
        var output = FeedForward(inputData).Outputs[^1];

        if (_cost is CrossEntropyCost)
            TensorPrimitives.SoftMax(output, output);

        return output;
    }


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

                outputs[iLayer][iNeuron] = _activation.Compute(weightedSum);
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

            stopwatch.Stop();
            var (accuracy, cost) = Evaluate(_cost, validationData, regularization);

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

        var costs = _cost.Gradient(outputs[^1], expected, _activation.Gradient(weightedSum[^1]));

        costsBias[^1] = costs;
        costsWeights[^1] = costs.Multiply(weightedSum[^2].Transpose());

        for (var iLayer = Layers.Length - 2; iLayer > 0; iLayer--)
        {
            costs = Weights[iLayer + 1]
                .Transpose()
                .Multiply(costs)
                .Multiply(_activation.Gradient(weightedSum[iLayer]));

            costsBias[iLayer] = costs;
            costsWeights[iLayer] = costs.Multiply(outputs[iLayer - 1].Transpose());
        };

        return (costsBias, costsWeights);
    }
}
