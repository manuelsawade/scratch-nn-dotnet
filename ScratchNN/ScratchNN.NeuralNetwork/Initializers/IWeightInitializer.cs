namespace ScratchNN.NeuralNetwork.Initializers;

public interface IWeightInitializer
{
    float Initialize(Random random, int inputNeurons);
}
