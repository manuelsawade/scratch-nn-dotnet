namespace ScratchNN.NeuralNetwork.Activations;

public static class SigmoidActivation
{
    public static float[] Steepness(float[] input)
    {
        return input.Select(SigmoidDerivative).ToArray();
    }

    public static float[] Function(float[] input)
    {
        return input.Select(Sigmoid).ToArray();
    }

    public static float Function(float input)
    {
        return Sigmoid(input);
    }

    private static float Sigmoid(float input)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-input));
    }

    private static float SigmoidDerivative(float input)
    {
        return Sigmoid(input) * (1 - Sigmoid(input));
    }
}
