using ScratchNN.NeuralNetwork.Extensions;
using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork.CostFunctions;

public class QuadraticCost : ICostFunction
{
    public float Computation(float[] output, float[] expected)
    {
        return (float)(0.5 * Math.Pow(TensorPrimitives.Norm(output.Subtract(expected)), 2.0));
    }

    public float[] Cost(float[] output, float[] expected, float[] weightedSum)
    {
        return output.Subtract(expected).Multiply(weightedSum);
    }
}
