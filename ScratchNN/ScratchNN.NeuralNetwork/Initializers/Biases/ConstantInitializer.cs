namespace ScratchNN.NeuralNetwork.Initializers.Biases;

public class ConstantInitializer(float value = 0.01f) : IBiasInitializer
{
    public float Initialize(Random random) => value;
}
