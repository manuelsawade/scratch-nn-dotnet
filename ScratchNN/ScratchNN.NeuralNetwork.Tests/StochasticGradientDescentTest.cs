using FluentAssertions;

using LabeledData = (float[] InputData, float[] ExpectedData);

namespace ScratchNN.NeuralNetwork.Tests;

public class StochasticGradientDescentTest
{
    private int[] _networkLayer;

    private float[][] _biases;
    private float[][][] _weights;

    private LabeledData[] _data;
    private float _learningRate;
    private float[][] _expected;

    [SetUp]
    public void Setup()
    {
        _networkLayer = [4, 5, 3];

        _biases = [
            [],
            [0.5f, 0.25f, -0.2f, 0.3f, -0.5f],
            [0.05f, -0.0325f, 0.35f],
        ];

        _weights = [
            [],
            [
                [0.38f, 0.123f, -0.341f, -0.2386f],
                [-0.123f, -0.4286f, -0.00324f, 0.23f],
                [0.00325f, 0.235f, -0.125f, 0.397f],
                [-0.388f, 0.495f, 0.173f, 0.298f],
                [-0.0023f, 0.153f, 0.222f, -0.387f]
            ],
            [
                [-0.023f, 0.142f, 0.3732f, 0.0052f, 0.09754f],
                [0.143f, 0.075f, -0.2755f, -0.5f, -0.5f],
                [0.0046f, -0.235f, -0.00015f, 0.125f, 0.345f]
            ]
        ];

        _data = [
            ([1, 2, 3, 4], [5, 6, 9]),
            ([2, 3, 4, 5], [1, 3, 4]),
            ([1, 8, 7, 6], [5, 7, 8]),
            ([1, 9, 3, 4], [2, 5, 9]),
            ([1, 2, 3, 4], [5, 6, 9]),
            ([2, 3, 4, 5], [1, 3, 4]),
            ([1, 8, 7, 6], [5, 7, 8]),
            ([1, 9, 3, 4], [2, 5, 9]),
        ];

        _learningRate = 0.5f;

        _expected = [
            [0.796827137f, 0.6727337f,   0.947449267f],
            [0.80219835f,  0.669064045f, 0.952524841f],
            [0.815240443f, 0.6279797f,   0.9648214f],
            [0.8170456f,   0.6555715f,   0.967523634f],
            [0.796827137f, 0.6727337f,   0.947449267f],
            [0.80219835f,  0.669064045f, 0.952524841f],
            [0.815240443f, 0.6279797f,   0.9648214f],
            [0.8170456f,   0.6555715f,   0.967523634f],
        ];
    }

    [Test]
    public void SimpleNeuralNetwork_should_calculate_stochastic_gradient_descent()
    {
        var sut = new SimpleNeuralNetwork(
            _networkLayer,
            _biases,
            _weights);

        sut.UpdateParameters(_data, _learningRate);

        var yPredicted0 = sut.Predict(_data[0].InputData);
        var yPredicted1 = sut.Predict(_data[1].InputData);
        var yPredicted2 = sut.Predict(_data[2].InputData);
        var yPredicted3 = sut.Predict(_data[3].InputData);
        var yPredicted4 = sut.Predict(_data[4].InputData);
        var yPredicted5 = sut.Predict(_data[5].InputData);
        var yPredicted6 = sut.Predict(_data[6].InputData);
        var yPredicted7 = sut.Predict(_data[7].InputData);

        Assert_Predictions(yPredicted0, _expected[0]);
        Assert_Predictions(yPredicted1, _expected[1]);
        Assert_Predictions(yPredicted2, _expected[2]);
        Assert_Predictions(yPredicted3, _expected[3]);
        Assert_Predictions(yPredicted4, _expected[4]);
        Assert_Predictions(yPredicted5, _expected[5]);
        Assert_Predictions(yPredicted6, _expected[6]);
        Assert_Predictions(yPredicted7, _expected[7]);
    }
    
    private static void Assert_Predictions(float[] predictedValues, float[] expectedValues)
    {
        foreach (var (predicted, expected) in predictedValues.Zip(expectedValues))
        {
            predicted.Should().Be(expected);
        }
    }
}