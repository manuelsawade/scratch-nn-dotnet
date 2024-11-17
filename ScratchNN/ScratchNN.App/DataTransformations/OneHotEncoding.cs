namespace ScratchNN.App.DataTransformations;

public class OneHotEncoding
{
    public static float[][] Transform<TType>(TType[] inputs)
    {
        var distinctedInputs = inputs.Distinct().Order().ToArray();

        return inputs
            .Select(input => Transform(input, distinctedInputs))
            .ToArray();
    }

    private static float[] Transform<TType>(TType input, TType[] distinctedInputs)
    {
        return distinctedInputs
            .Select(i => EqualityComparer<TType>.Default.Equals(i, input) ? 1f : 0f)
            .ToArray();
    }
}
