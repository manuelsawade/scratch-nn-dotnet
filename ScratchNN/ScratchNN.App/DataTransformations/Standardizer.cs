namespace ScratchNN.App.DataTransformations;

public class Standardizer
{
    public static float[][] Transform(float[][] inputs)
    {
        var flattenedInputs = inputs.SelectMany(input => input);

        var inputsAverage = flattenedInputs.Average();

        var inputsStandardDeviation = (float)Math.Sqrt(
            flattenedInputs.Average(value => Math.Pow(value - inputsAverage, 2)));

        return inputs
            .Select(input => input
                .Select(value => value - inputsAverage / inputsStandardDeviation)
                .ToArray())
            .ToArray();
    }
}
