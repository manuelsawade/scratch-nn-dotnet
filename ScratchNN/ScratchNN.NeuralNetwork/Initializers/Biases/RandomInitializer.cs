using ScratchNN.NeuralNetwork.Extensions;

namespace ScratchNN.NeuralNetwork.Initializers.Biases;

public class RandomInitializer : IBiasInitializer
{
    public float Initialize(Random random)
    {
        return random.GetRandomGaussianVariable(0, 1);
    }
}