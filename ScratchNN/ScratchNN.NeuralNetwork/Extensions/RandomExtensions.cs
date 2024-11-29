namespace ScratchNN.NeuralNetwork.Extensions;

public static class RandomExtensions
{
    public static float GetRandomGaussianVariable(this Random random, int mean, int standardDeviation)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        var randStdNormal =
            Math.Sqrt(-2.0 * Math.Log(u1)) *
            Math.Sin(2.0 * Math.PI * u2);

        return (float)(mean + standardDeviation * randStdNormal);
    }

    public static float GetRandomGaussianVariable(this Random random, int mean, float standardDeviation)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        var randStdNormal = 
            MathF.Sqrt(-2.0f * MathF.Log((float)u1)) * 
            MathF.Sin(2.0f * MathF.PI * (float)u2);

        return mean + standardDeviation * randStdNormal;
    }
}
