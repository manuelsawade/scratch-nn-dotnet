namespace ScratchNN.NeuralNetwork.CostFunctions;

public interface ICostFunction
{
    float[] Gradient(float[] output, float[] expected, float[] weightedSum);

    float Compute(float[] output, float[] expected);
}
