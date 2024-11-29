using ScratchNN.NeuralNetwork.Extensions;

namespace ScratchNN.NeuralNetwork.Initializers.Weights;

public class XavierInitializer : IWeightInitializer
{
    public float Initialize(Random random, int inputNeurons)
    {
        return random.GetRandomGaussianVariable(0, 1) / (float)Math.Sqrt(inputNeurons);
    }
}