namespace ScratchNN.NeuralNetwork.Activations;

public class RectifiedLinearUnitActivation
{
    public static float[] Function(float[] input)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = Function(input[i]);

        return input;
    }

    public static float Function(float input)
    {
        return (input < 0) ? 0 : input;
    }

    public static float[] Derivate(float[] input)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = Derivate(input[i]);

        return input;
    }

    public static float Derivate(float input)
    {
        return input >= 0 ? 1 : 0;
    }
}
