namespace ScratchNN.NeuralNetwork.Activations;

public class RectifiedLinearUnitActivation : IActivationFunction
{
    public float[] Compute(float[] input)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = Compute(input[i]);

        return input;
    }

    public float Compute(float input)
    {
        return (input < 0) ? 0 : input;
    }

    public float[] Gradient(float[] input)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = Gradient(input[i]);

        return input;
    }

    public float Gradient(float input)
    {
        return input >= 0 ? 1 : 0;
    }
}
