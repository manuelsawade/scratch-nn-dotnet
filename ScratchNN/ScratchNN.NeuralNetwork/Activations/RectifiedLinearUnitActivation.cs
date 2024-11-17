namespace ScratchNN.NeuralNetwork.Activations;

public class RectifiedLinearUnitActivation
{
    public static float[] Function(float[] input)
    {
        return input.Select(Function).ToArray();
    }

    private static float Function(float input)
    {
        return (input < 0) ? 0 : input;
    }

    public float[] Derivate(float[] input)
    {
        throw new NotImplementedException();
    }
}
