using ScratchNN.NeuralNetwork.Extensions;
using System.Numerics.Tensors;

namespace ScratchNN.NeuralNetwork.Activations;

public class SigmoidActivation : IActivationFunction
{
    public float Activation(float input)
    {
        float[] activated = [0f];
        TensorPrimitives.Sigmoid([input], activated);

        return activated[0];
    }

    public float[] Activation(float[] input)
    {
        var activated = input.Length.New<float>();
        TensorPrimitives.Sigmoid(input, activated);

        return activated;
    }

    public float Gradient(float input)
    {
        var activation = Activation(input);
        return activation * (1 - activation);
    }

    public float[] Gradient(float[] input)
    {
        ReadOnlySpan<float> activation = Activation(input);
        return activation.Multiply(activation.OneSubtract()).ToArray();
    }
}
