namespace ScratchNN.NeuralNetwork.Extensions;

internal static class ArrayExtensions
{
    private static readonly Random rng = new();

    internal static T[] Shuffle<T>(this T[] source, Random? random = null)
    {
        random ??= rng;

        int n = source.Length;
        while (n > 1)
        {
            n--;
            int k = random.Next(n + 1);
            (source[n], source[k]) = (source[k], source[n]);
        }

        return source;
    }

    public static int Shape<TType>(this TType[] source)
    {
        return source.Length;
    }

    public static int[] Shape<TType>(this TType[][] source)
    {
        return source.Select(Shape).ToArray();
    }

    public static int[][] Shape<TType>(this TType[][][] source)
    {
        return source.Select(Shape).ToArray();
    }

    public static TType[] New<TType>(this int source)
    {
        return new TType[source];
    }

    public static TType[][] New<TType>(this int[] source)
    {
        return source.Select(New<TType>).ToArray();
    }

    public static TType[][][] New<TType>(this int[][] source)
    {
        return source.Select(New<TType>).ToArray();
    }

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
            for (var j = 0; j < a[i].Length; j++)
                a[i][j] = a[i][j] + b[i][j];

        return a;
    }

    public static float[][][] Add(this float[][][] a, float[][][] b)
    {
        for (var i = 0; i < a.Length; i++)
            for (var j = 0; j < a[i].Length; j++)
                for (var k = 0; k < a[i][j].Length; k++)
                    a[i][j][k] = a[i][j][k] + b[i][j][k];

        return a;
    }

    public static float[] Substract(this float[] a, float[] b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] = a[i] - b[i];

        return a;
    }

    public static float[][] Substract(this float[][] a, float[][] b)
    {
        for (var i = 0; i < a.Length; i++)
            for (var j = 0; j < a[i].Length; j++)
                a[i][j] = a[i][j] - b[i][j];

        return a;
    }

    public static TType[][] Transpose<TType>(this TType[] source)
    {
        var result = new TType[source.Length][];

        for (var i = 0; i < source.Length; i++)
        {
            result[i] = [source[i]];
        }

        return result;
    }

    public static TType[][] Transpose<TType>(this TType[][] source)
    {
        var result = new TType[source[0].Length]
            .Select(r => new TType[source.Length])
            .ToArray();

        for (int i = 0; i < source.Length; i++)
        {
            for (int j = 0; j < source[i].Length; j++)
            {
                result[j][i] = source[i][j];
            }
        }

        return result;
    }
}
