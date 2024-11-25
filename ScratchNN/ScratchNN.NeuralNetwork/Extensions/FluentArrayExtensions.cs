namespace ScratchNN.NeuralNetwork.Extensions;

internal static class FluentArrayExtensions
{   
    public static float[][] Multiply(this float[] a, float[][] b)
    {
        var result = new float[a.Length][];

        for (var i = 0; i < a.Length; i++)
        {
            result[i] = new float[b.Length];
            for (var j = 0; j < b.Length; j++)
            {
                result[i][j] = a[i] * b[j][0];
            }
        }

        return result;
    }

    public static float[] Multiply(this float[][] a, float[] b)
    {
        var result = new float[a.Length];

        for (var rowA = 0; rowA < a.Length; rowA++)
        {
            for (var colA = 0; colA < a[rowA].Length; colA++)
            {
                result[rowA] += a[rowA][colA] * b[colA];
            }
        }

        return result;
    }
    public static float[] Multiply(this float[] a, float[] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] = a[i] * b[i];

        return a;
    }

    public static float[] Add(this float[] a, float[] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] = a[i] + b[i];

        return a;
    }

    public static float[][] Add(this float[][] a, float[][] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i].Add(b[i]);

        return a;
    }

    public static float[][][] Add(this float[][][] a, float[][][] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i].Add(b[i]);

        return a;
    }

    public static float[] Subtract(this float[] a, float[] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] = a[i] - b[i];

        return a;
    }

    public static float[][] Subtract(this float[][] a, float[][] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i].Subtract(b[i]);

        return a;
    }

    public static float[] ClampEpsilon(this float[] a)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] = Math.Clamp(a[i], float.Epsilon, 1 - float.Epsilon);

        return a;
    }
}
