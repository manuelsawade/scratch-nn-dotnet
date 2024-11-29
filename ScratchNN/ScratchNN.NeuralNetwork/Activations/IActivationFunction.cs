namespace ScratchNN.NeuralNetwork.Activations;

public interface IActivationFunction
{
    public float Compute(float weightedSums);
    public float[] Compute(float[] weightedSums);
    public float Gradient(float weightedSums);
    public float[] Gradient(float[] weightedSums);
}
