using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork.Extensions;

public static class FluentTensorExtensions
{
    public static float[] Subtract(this ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var result = new float[a.Length];
        TensorPrimitives.Subtract(a, b, result);
        return result;
    }

    public static ReadOnlySpan<float> OneSubtract(this ReadOnlySpan<float> a)
    {
        var result = new float[a.Length];
        TensorPrimitives.Subtract(1, a, result);
        return result;
    }

    public static ReadOnlySpan<float> Log(this ReadOnlySpan<float> a)
    {        
        var result = new float[a.Length];
        TensorPrimitives.Log(a.ToArray().ClampEpsilon(), result);
        return result;
    }

    public static ReadOnlySpan<float> Multiply(this ReadOnlySpan<float> a, Span<float> b)
    {
        TensorPrimitives.Multiply(a, b, b);
        return b;
    }

    public static ReadOnlySpan<float> Multiply(this ReadOnlySpan<float> a, float b)
    {
        var result = new float[a.Length];
        TensorPrimitives.Multiply(a, b, result);
        return result;
    }

    public static ReadOnlySpan<float> Multiply(this ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var result = a.Length.New<float>();
        TensorPrimitives.Multiply(a, b, result);
        return result;
    }

    public static ReadOnlySpan<float> Multiply(this float[] a, float b)
    {
        var result = a.Length.New<float>();
        TensorPrimitives.Multiply(a, b, result);
        return result;
    }

    public static ReadOnlySpan<float> Divide(this ReadOnlySpan<float> a, float b)
    {
        var result = new float[a.Length];
        TensorPrimitives.Divide(a, b, result);
        return result;
    }

    public static ReadOnlySpan<float> Negate(this ReadOnlySpan<float> a)
    {
        var result = new float[a.Length];
        TensorPrimitives.Negate(a, result);
        return result;
    }

    public static float Norm(this float[][][] a)
    {
        return a
            .SelectMany(neuron => neuron
                .Select(weights => TensorPrimitives.Norm(weights)))
            .Sum();
    }
}
