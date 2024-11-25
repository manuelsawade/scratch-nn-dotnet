using ScratchNN.NeuralNetwork.Extensions;

namespace ScratchNN.NeuralNetwork;

public abstract class NeuralNetworkBase
{
    public virtual int Seed { get; init; }

    public virtual int[] Layers { get; init; } = [];

    public virtual float[][] Biases { get; init; } = [];

    public virtual float[][][] Weights { get; init; } = [];

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

    protected static float[][] InitBiases(int[] layers, Random random)
    {
        var allBiases = new float[layers.Length][];
        allBiases[0] = [];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerBiases = new float[layers[iLayer]];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                layerBiases[iNeuron] = Initialize(random);
            }

            allBiases[iLayer] = layerBiases;
        }

        return allBiases;
    }

    protected static float[][][] InitWeights(int[] layers, Random random)
    {
        var allWeights = new float[layers.Length][][];
        allWeights[0] = [];

        for (var iLayer = 1; iLayer < layers.Length; iLayer++)
        {
            var layerWeights = new float[layers[iLayer]][];
            var previousNeurons = layers[iLayer - 1];

            for (var iNeuron = 0; iNeuron < layers[iLayer]; iNeuron++)
            {
                var neuronWeights = new float[previousNeurons];

                for (var iWeight = 0; iWeight < previousNeurons; iWeight++)
                {
                    neuronWeights[iWeight] = Initialize(random, respectiveTo: previousNeurons);
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
