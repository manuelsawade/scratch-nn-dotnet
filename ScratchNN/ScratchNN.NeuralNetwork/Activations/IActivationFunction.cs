namespace ScratchNN.NeuralNetwork.Activations;

public interface IActivationFunction
{
    public float Activation(float weightedSums);
    public float[] Activation(float[] weightedSums);
    public float Gradient(float weightedSums);
    public float[] Gradient(float[] weightedSums);
}
