namespace ScratchNN.NeuralNetwork.Extensions;

internal static class RandomExtensions
{
    internal static float GetRandomGaussianVariable(this Random random, int mean, int standardDeviation)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        var randStdNormal =
            Math.Sqrt(-2.0 * System.Math.Log(u1)) *
            Math.Sin(2.0 * System.Math.PI * u2);

        return (float)(mean + standardDeviation * randStdNormal);
    }
}
