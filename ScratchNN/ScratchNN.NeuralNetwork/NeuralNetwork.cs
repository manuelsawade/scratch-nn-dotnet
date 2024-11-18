using ML.NeuralNetwork.Extensions;
using ScratchNN.NeuralNetwork.Activations;
using ScratchNN.NeuralNetwork.Extensions;
using System.Diagnostics;
using System.Numerics.Tensors;
using static ScratchNN.NeuralNetwork.Extensions.ArrayExtensions;

namespace ScratchNN.NeuralNetwork;

using LabeledData = (float[] InputData, float[] Expected);

public class NeuralNetwork
{
    private readonly static Random _random = new();

    private readonly int[] _layers;
    private readonly float[][] _biases;
    private readonly float[][][] _weights;

    public NeuralNetwork(int[] layers)
    {
        _layers = layers;
        _biases = InitBiases(layers);
        _weights = InitWeights(layers);
    }

    public NeuralNetwork(
        int[] layers,
        float[][] biases,
        float[][][] weights)
    {
        _layers = layers;
        _biases = biases;
        _weights = weights;
    }

    public float[] Predict(float[] inputData) => FeedForward(inputData).Outputs[^1];


    public (float[][] Outputs, float[][] WeightedSums) FeedForward(float[] inputData)
    {
        var outputs = _layers.New<float>();
        outputs[0] = inputData;

        var weightedSums = _layers.New<float>();

        for (var iLayer = 1; iLayer < _layers.Length; iLayer++)
        {
            for (var iNeuron = 0; iNeuron < _layers[iLayer]; iNeuron++)
            {
                var inputs = outputs[iLayer - 1];
                var weights = _weights[iLayer][iNeuron];
                var bias = _biases[iLayer][iNeuron];
                var weightedInput = 0f;

                for (var iInput = 0; iInput < inputs.Length; iInput++)
                {
                    weightedInput += inputs[iInput] * weights[iInput];
                }

                var weightedSum = weightedInput + bias;

                outputs[iLayer][iNeuron] = Activation(weightedSum);
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

            for (var iBatch = 0; iBatch < miniBatches.Length; iBatch++)
            {
                Console.Write($"Epoch {epoch,2} | Fit Batches: {iBatch}/{miniBatches.Length}");
                UpdateParameters(miniBatches[iBatch], learningRate, regularization);

                ConsoleExtensions.ClearCurrentLine();
            }

            Evaluate(validationData);
        }
    }

    public void UpdateParameters(LabeledData[] trainingBatch, float learningRate, float regularization)
    {
        var costSumBias = _biases[1..].Shape().New<float>();
        var costSumWeights = _weights[1..].Shape().New<float>();

        foreach (var (InputData, Expected) in trainingBatch)
        {
            var (costsBias, costsWeights) = Backpropagation(InputData, Expected);

            costSumBias = costSumBias.Add(costsBias);
            costSumWeights = costSumWeights.Add(costsWeights);
        }

        IterateNetwork(
            biasAction: (iLayer, iNeuron, bias) =>
            {
                _biases[iLayer][iNeuron] =
                    bias - learningRate * costSumBias[iLayer - 1][iNeuron] / trainingBatch.Length;
            },
            weightAction: (iLayer, iNeuron, iWeight, weight) =>
            {
                _weights[iLayer][iNeuron][iWeight] =
                    (1 - learningRate * (regularization / trainingBatch.Length)) *
                    weight - learningRate * costSumWeights[iLayer - 1][iNeuron][iWeight] / trainingBatch.Length;
            });
    }

    public (float[][], float[][][]) Backpropagation(float[] inputData, float[] expected)
    {
        var costsBias = _biases[1..].Shape().New<float>();
        var costsWeights = _weights[1..].Shape().New<float>();

        var (outputs, weightedSum) = FeedForward(inputData);

        var costs = outputs[^1]
            .Subtract(expected)
            .Multiply(ActivationSteepness(weightedSum[^1]));

        costsBias[^1] = costs;
        costsWeights[^1] = costs.Multiply(weightedSum[^2].Transpose());

        for (var iLayer = _layers.Length - 2; iLayer > 0; iLayer--)
        {
            costs = _weights[iLayer + 1]
                .Transpose()
                .Multiply(costs)
                .Multiply(ActivationSteepness(weightedSum[iLayer]));

            costsBias[iLayer - 1] = costs;
            costsWeights[iLayer - 1] = costs.Multiply(outputs[iLayer - 1].Transpose());
        };

        return (costsBias, costsWeights);
    }

    public void Evaluate(LabeledData[] labeledTestData)
    {
        var correctPredictions = 0;
        var costs = new float[labeledTestData.Length];

        Parallel.For(0, labeledTestData.Length, new() { MaxDegreeOfParallelism = 1 }, (iData) =>
        {
            var (InputData, Expected) = labeledTestData[iData];
            var expectedLabel = Array.FindIndex(Expected, label => label == 1);

            var predicted = Predict(InputData);
            var expected = predicted[expectedLabel];

            var delta = new float[predicted.Length];
            TensorPrimitives.Subtract(Expected, predicted, delta);

            costs[iData] = (float)(0.5 * System.Math.Pow((double)TensorPrimitives.Norm(delta), 2.0)) / labeledTestData.Length;

            if (expected == predicted.Max())
            {
                Interlocked.Increment(ref correctPredictions);
            }
        });

        var accuracy = (float)System.Math.Round(correctPredictions / (double)labeledTestData.Length, 2);

        var normedWeights = _weights[1..].SelectMany(neuron => neuron.Select(weights => TensorPrimitives.Norm(weights))).Sum();
        var completeCost = 0.5f * (1 / labeledTestData.Length) * normedWeights + costs.Sum();

        Console.WriteLine($"{labeledTestData.Length,5}/{correctPredictions} = Accuracy: {accuracy,-4} | Cost: {completeCost}");
    }

    private static float[][] InitBiases(int[] layers)
    {
        var allBiases = new float[layers.Length][];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerBiases = new float[layers[iLayer]];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                layerBiases[iNeuron] = Initialize(_random);
            }

            allBiases[iLayer] = layerBiases;
        }

        return allBiases;
    }

    private static float[][][] InitWeights(int[] layers)
    {
        var allWeights = new float[layers.Length][][];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerWeights = new float[layers[iLayer]][];
            var previousNeurons = layers[iLayer - 1];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                var neuronWeights = new float[previousNeurons];

                for (var iWeight = 0; iWeight < previousNeurons; iWeight++)
                {
                    neuronWeights[iWeight] = Initialize(_random, respectiveTo: previousNeurons);
                }

                layerWeights[iNeuron] = neuronWeights;
            }

            allWeights[iLayer] = layerWeights;
        }

        return allWeights;
    }

    private static float Initialize(Random random, int? respectiveTo = null)
    {
        var value = random.GetRandomGaussianVariable(0, 1);
        return respectiveTo.HasValue
            ? value / (float)Math.Sqrt(respectiveTo.Value)
            : value;
    }

    private static float[] Activation(float[] weightedSums)
    {
        return SigmoidActivation.Function(weightedSums);
    }

    private static float Activation(float weightedSums)
    {
        return SigmoidActivation.Function(weightedSums);
    }

    private static float[] ActivationSteepness(float[] weightedSums)
    {
        return SigmoidActivation.Steepness(weightedSums);
    }

    private void IterateNetwork(
        Action<int, int, float> biasAction,
        Action<int, int, int, float> weightAction)
    {
        for (var iLayer = 1; iLayer < _layers.Length - 1; iLayer++)
        {
            for (var iNeuron = 0; iNeuron < _layers.Length - 1; iNeuron++)
            {
                biasAction(iLayer, iNeuron, _biases[iLayer][iNeuron]);

                for (var iWeight = 0; iWeight < _weights[iLayer][iNeuron].Length; iWeight++)
                {
                    weightAction(iLayer, iNeuron, iWeight, _weights[iLayer][iNeuron][iWeight]);
                }
            }
        }
    }
}
