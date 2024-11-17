namespace ScratchNN.NeuralNetwork.Activations;

public class SigmoidActivation
{
    public static float[] Steepness(float[] input)
    {
        return input.Select(SigmoidDerivate).ToArray();
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

    private static float SigmoidDerivate(float input)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-input));
    }
}
