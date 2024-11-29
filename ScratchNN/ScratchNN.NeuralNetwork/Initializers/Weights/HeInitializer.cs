using ScratchNN.NeuralNetwork.Extensions;

namespace ScratchNN.NeuralNetwork.Initializers.Weights;

public class HeInitializer : IWeightInitializer
{
    public float Initialize(Random random, int inputNeurons)
    {
        return random.GetRandomGaussianVariable(0, (float)Math.Sqrt(2f / inputNeurons));
    }
}
