﻿using ScratchNN.NeuralNetwork.CostFunctions;
using ScratchNN.NeuralNetwork.Extensions;
using ScratchNN.NeuralNetwork.Initializers.Biases;
using ScratchNN.NeuralNetwork.Initializers.Weights;
using System.Diagnostics;

namespace ScratchNN.NeuralNetwork.Implementations;

public class SimpleNeuralNetwork : NeuralNetworkBase
{
    private readonly Random _random = new();

    public override int Seed { get; init; }
    public override int[] Layers { get; init; }
    public override float[][] Biases { get; init; }
    public override float[][][] Weights { get; init; }

    public SimpleNeuralNetwork(int[] layers, int? seed = null)
    {
        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = InitBiases(layers, _random, new RandomInitializer());
        Weights = InitWeights(layers, _random, new XavierInitializer());
    }

    public SimpleNeuralNetwork(
        int[] layers,
        float[][] biases,
        float[][][] weights,
        int? seed = null)
    {
        (_random, Seed) = InitRandom(seed);

        Layers = layers;
        Biases = biases;
        Weights = weights;
    }

    public override float[] Predict(float[] inputData) => FeedForward(inputData).Outputs[^1];


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
        float learningRate)
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
                UpdateParameters(miniBatches[iBatch], learningRate);

                ConsoleExtensions.ClearCurrentLine();
            }

            stopwatch.Stop();
            var (accuracy, cost) = Evaluate(new QuadraticCost(), validationData, 1);

            Console.WriteLine($"Accuracy: {accuracy,-4} | Cost: {cost,-6} | Elapsed: {stopwatch.Elapsed}");
        }
    }

    public void UpdateParameters(LabeledData[] trainingBatch, float learningRate)
    {
        var costSumBias = Biases.Shape().New<float>();
        var costSumWeights = Weights.Shape().New<float>();

        foreach (var (inputData, expected) in trainingBatch)
        {
            var (costsBias, costsWeights) = Backpropagation(inputData, expected);

            costSumBias = costSumBias.Add(costsBias);
            costSumWeights = costSumWeights.Add(costsWeights);
        }

        IterateNetwork(
            biasAction: (iLayer, iNeuron, bias) =>
            {
                Biases[iLayer][iNeuron] = bias - learningRate * costSumBias[iLayer][iNeuron] / trainingBatch.Length;
            },
            weightAction: (iLayer, iNeuron, iWeight, weight) =>
            {
                Weights[iLayer][iNeuron][iWeight] = weight - learningRate * costSumWeights[iLayer][iNeuron][iWeight] / trainingBatch.Length;
            });
    }

    public (float[][], float[][][]) Backpropagation(float[] inputData, float[] expected)
    {
        var costsBias = Biases.Shape().New<float>();
        var costsWeights = Weights.Shape().New<float>();

        var (outputs, weightedSum) = FeedForward(inputData);

        var costs = outputs[^1].Subtract(expected)
            .Multiply(ActivationSteepness(weightedSum[^1]));

        costsBias[^1] = costs;
        costsWeights[^1] = costs.Multiply(outputs[^2].Transpose());

        for (var iLayer = Layers.Length - 2; iLayer > 0; iLayer--)
        {
            costs = Weights[iLayer + 1]
                .Transpose()
                .Multiply(costs)
                .Multiply(ActivationSteepness(weightedSum[iLayer]));

            costsBias[iLayer] = costs;
            costsWeights[iLayer] = costs.Multiply(outputs[iLayer - 1].Transpose());
        };

        return (costsBias, costsWeights);
    }

    private static float Activation(float weightedSums)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-weightedSums));
    }

    private static float[] ActivationSteepness(float[] weightedSums)
    {
        for (var i = 0; i < weightedSums.Length; i++)
        {
            var activation = Activation(weightedSums[i]);
            weightedSums[i] = activation * (1 - activation);
        }

        return weightedSums;
    }
}
