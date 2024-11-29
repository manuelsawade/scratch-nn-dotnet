using ScratchNN.NeuralNetwork.CostFunctions;
using ScratchNN.NeuralNetwork.Extensions;
using ScratchNN.NeuralNetwork.Initializers;
using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork;

public abstract class NeuralNetworkBase
{
    public abstract int Seed { get; init; }

    public abstract int[] Layers { get; init; }

    public abstract float[][] Biases { get; init; }

    public abstract float[][][] Weights { get; init; }

    public abstract float[] Predict(float[] inputData);

    public (float Accuracy, float Cost) Evaluate(
        ICostFunction costFunction, 
        LabeledData[] labeledTestData, 
        float regularization)
    {
        var correctPredictions = 0;
        var costs = new float[labeledTestData.Length];

        Parallel.For(
            0, 
            labeledTestData.Length, 
            new() { MaxDegreeOfParallelism = 10 }, 
            (iData) =>
            {
                var (inputData, expected) = labeledTestData[iData];
                var output = Predict(inputData);

                if (TensorPrimitives.IndexOfMax(output) == TensorPrimitives.IndexOfMax(expected))
                {
                    Interlocked.Increment(ref correctPredictions);
                }

                costs[iData] = costFunction.Compute(output, expected) / labeledTestData.Length;
            });

        var accuracy = (float)Math.Round(correctPredictions / (double)labeledTestData.Length, 2);
        var completeCost = MathF.Round(0.5f * (regularization / labeledTestData.Length) * Weights[1..].Norm() + costs.Sum(), 4);

        return (accuracy, completeCost);
    }

    public float EvaluateAccuracy(
        LabeledData[] labeledTestData)
    {
        var correctPredictions = 0;
        var costs = new float[labeledTestData.Length];

        Parallel.For(
            0,
            labeledTestData.Length,
            new() { MaxDegreeOfParallelism = 10 },
            (iData) =>
            {
                var (inputData, expected) = labeledTestData[iData];
                var output = Predict(inputData);

                if (TensorPrimitives.IndexOfMax(output) == TensorPrimitives.IndexOfMax(expected))
                {
                    Interlocked.Increment(ref correctPredictions);
                }
            });

        return (float)Math.Round(correctPredictions / (double)labeledTestData.Length, 2);
    }


    protected void IterateNetwork(
        Action<int, int, float> biasAction,
        Action<int, int, int, float> weightAction)
    {
        for (var iLayer = 1; iLayer < Layers.Length; iLayer++)
        {
            for (var iNeuron = 0; iNeuron < Layers[iLayer]; iNeuron++)
            {
                biasAction(iLayer, iNeuron, Biases[iLayer][iNeuron]);

                for (var iWeight = 0; iWeight < Weights[iLayer][iNeuron].Length; iWeight++)
                {
                    weightAction(iLayer, iNeuron, iWeight, Weights[iLayer][iNeuron][iWeight]);
                }
            }
        }
    }

    protected void IterateNetwork(
        Action<int, float[]> biasAction,
        Action<int, int, float[]> weightAction)
    {
        for (var iLayer = 1; iLayer < Layers.Length; iLayer++)
        {
            biasAction(iLayer, Biases[iLayer]);

            for (var iNeuron = 0; iNeuron < Layers[iLayer]; iNeuron++)
            {               
                weightAction(iLayer, iNeuron, Weights[iLayer][iNeuron]);
            }
        }
    }

    protected static (Random, int) InitRandom(int? seed)
    {
        if (!seed.HasValue)
            seed = new Random().Next();

        return (new Random(seed.Value), seed.Value);
    }

    protected static float[][] InitBiases(int[] layers, Random random, IBiasInitializer biasInitializer)
    {
        var allBiases = new float[layers.Length][];
        allBiases[0] = [];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerBiases = new float[layers[iLayer]];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                layerBiases[iNeuron] = biasInitializer.Initialize(random);
            }

            allBiases[iLayer] = layerBiases;
        }

        return allBiases;
    }

    protected static float[][][] InitWeights(int[] layers, Random random, IWeightInitializer weightInitializer)
    {
        var allWeights = new float[layers.Length][][];
        allWeights[0] = [];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerWeights = new float[layers[iLayer]][];
            var inputNeurons = layers[iLayer - 1];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                var neuronWeights = new float[inputNeurons];

                for (var iWeight = 0; iWeight < inputNeurons; iWeight++)
                {
                    neuronWeights[iWeight] = weightInitializer.Initialize(random, inputNeurons);
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
}
