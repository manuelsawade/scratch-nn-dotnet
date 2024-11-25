namespace ScratchNN.NeuralNetwork.CostFunctions;

public interface ICostFunction
{
    float[] Cost(float[] output, float[] expected, float[] weightedSum);

    float Computation(float[] output, float[] expected);
}
