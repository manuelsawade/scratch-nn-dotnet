using FluentAssertions;

namespace ScratchNN.NeuralNetwork.Tests;

public class StochasticGradientDescentTest
{
    private int[] _networkLayer;

    private float[][] _biases;
    private float[][][] _weights;

    private ModelData[] _data;
    private float _learningRate;
    private float[][] _expected;

    [SetUp]
    public void Setup()
    {
        _networkLayer = [4, 5, 3];

        _biases =
        [
            [0.5f, 0.25f, -0.2f, 0.3f, -0.5f],
            [0.05f, -0.0325f, 0.35f],
        ];

        _weights =
        [[
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
        ]];

        _data = [
            new ModelData { Features = [1, 2, 3, 4], Label = [5, 6, 9] },
            new ModelData { Features = [2, 3, 4, 5], Label = [1, 3, 4] },
            new ModelData { Features = [1, 8, 7, 6], Label = [5, 7, 8] },
            new ModelData { Features = [1, 9, 3, 4], Label = [2, 5, 9] },
            new ModelData { Features = [1, 2, 3, 4], Label = [5, 6, 9] },
            new ModelData { Features = [2, 3, 4, 5], Label = [1, 3, 4] },
            new ModelData { Features = [1, 8, 7, 6], Label = [5, 7, 8] },
            new ModelData { Features = [1, 9, 3, 4], Label = [2, 5, 9] },
        ];

        _learningRate = 0.5f;

        _expected = [
            [0.784782767f, 0.678221f, 0.9452476f],
            [0.788222f, 0.671789765f, 0.948840737f],
            [0.8024867f, 0.6300075f, 0.9601097f],
            [0.8059054f, 0.6566872f, 0.963045835f],
            [0.784782767f, 0.678221f, 0.9452476f],
            [0.788222f, 0.671789765f, 0.948840737f],
            [0.8024867f, 0.6300075f, 0.9601097f],
            [0.8059054f, 0.6566872f, 0.963045835f]
        ];
    }

    [Test]
    public void NeuralNetworkv0_should_calculate_stochastic_gradient_descent()
    {
        var sut = new NeuralNetwork(
            _networkLayer,
            _biases,
            _weights);

        sut.UpdateParameters(_data, 0.5f);

        var yPredicted0 = sut.Predict(_data[0].Features);
        var yPredicted1 = sut.Predict(_data[1].Features);
        var yPredicted2 = sut.Predict(_data[2].Features);
        var yPredicted3 = sut.Predict(_data[3].Features);
        var yPredicted4 = sut.Predict(_data[4].Features);
        var yPredicted5 = sut.Predict(_data[5].Features);
        var yPredicted6 = sut.Predict(_data[6].Features);
        var yPredicted7 = sut.Predict(_data[7].Features);

        Assert_Predictions(yPredicted0, _expected[0]);
        Assert_Predictions(yPredicted1, _expected[1]);
        Assert_Predictions(yPredicted2, _expected[2]);
        Assert_Predictions(yPredicted3, _expected[3]);
        Assert_Predictions(yPredicted4, _expected[4]);
        Assert_Predictions(yPredicted5, _expected[5]);
        Assert_Predictions(yPredicted6, _expected[6]);
        Assert_Predictions(yPredicted7, _expected[7]);
    }

    [Test]
    public void NeuralNetworkv1_should_calculate_stochastic_gradient_descent()
    {
        var sut = new NeuralNetworkv1(
            _networkLayer,
            (biases) => _biases,
            (weights) => _weights);

        sut.StochasticGradientDescent(_data, 0.5f);

        var yPredicted0 = sut.Predict(_data[0].Features);
        var yPredicted1 = sut.Predict(_data[1].Features);
        var yPredicted2 = sut.Predict(_data[2].Features);
        var yPredicted3 = sut.Predict(_data[3].Features);
        var yPredicted4 = sut.Predict(_data[4].Features);
        var yPredicted5 = sut.Predict(_data[5].Features);
        var yPredicted6 = sut.Predict(_data[6].Features);
        var yPredicted7 = sut.Predict(_data[7].Features);

        Assert_Predictions(yPredicted0, _expected[0]);
        Assert_Predictions(yPredicted1, _expected[1]);
        Assert_Predictions(yPredicted2, _expected[2]);
        Assert_Predictions(yPredicted3, _expected[3]);
        Assert_Predictions(yPredicted4, _expected[4]);
        Assert_Predictions(yPredicted5, _expected[5]);
        Assert_Predictions(yPredicted6, _expected[6]);
        Assert_Predictions(yPredicted7, _expected[7]);
    }

    [Test]
    public void NeuralNetworkv2_should_calculate_stochastic_gradient_descent()
    {
        var sut = new NeuralNetworkv2(
            _networkLayer,
            (biases) => _biases,
            (weights) => _weights);

        sut.StochasticGradientDescent(_data, 0.5f);

        var yPredicted0 = sut.Predict(_data[0].Features);
        var yPredicted1 = sut.Predict(_data[1].Features);
        var yPredicted2 = sut.Predict(_data[2].Features);
        var yPredicted3 = sut.Predict(_data[3].Features);
        var yPredicted4 = sut.Predict(_data[4].Features);
        var yPredicted5 = sut.Predict(_data[5].Features);
        var yPredicted6 = sut.Predict(_data[6].Features);
        var yPredicted7 = sut.Predict(_data[7].Features);

        Assert_Predictions(yPredicted0, _expected[0]);
        Assert_Predictions(yPredicted1, _expected[1]);
        Assert_Predictions(yPredicted2, _expected[2]);
        Assert_Predictions(yPredicted3, _expected[3]);
        Assert_Predictions(yPredicted4, _expected[4]);
        Assert_Predictions(yPredicted5, _expected[5]);
        Assert_Predictions(yPredicted6, _expected[6]);
        Assert_Predictions(yPredicted7, _expected[7]);
    }

    [Test]
    public void NeuralNetworkv3_should_calculate_stochastic_gradient_descent()
    {
        var sut = new NeuralNetworkv3(
            _networkLayer,
            (biases) => _biases,
            (weights) => _weights,
            new SigmoidActivation());

        sut.StochasticGradientDescent(_data, 0.5f);

        var yPredicted0 = sut.Predict(_data[0].Features);
        var yPredicted1 = sut.Predict(_data[1].Features);
        var yPredicted2 = sut.Predict(_data[2].Features);
        var yPredicted3 = sut.Predict(_data[3].Features);
        var yPredicted4 = sut.Predict(_data[4].Features);
        var yPredicted5 = sut.Predict(_data[5].Features);
        var yPredicted6 = sut.Predict(_data[6].Features);
        var yPredicted7 = sut.Predict(_data[7].Features);

        Assert_Predictions(yPredicted0, _expected[0]);
        Assert_Predictions(yPredicted1, _expected[1]);
        Assert_Predictions(yPredicted2, _expected[2]);
        Assert_Predictions(yPredicted3, _expected[3]);
        Assert_Predictions(yPredicted4, _expected[4]);
        Assert_Predictions(yPredicted5, _expected[5]);
        Assert_Predictions(yPredicted6, _expected[6]);
        Assert_Predictions(yPredicted7, _expected[7]);
    }


    [Test]
    public void NeuralNetworkv7_should_calculate_stochastic_gradient_descent()
    {
        static float[][] createEmpty()
        {
            var empty = new float[1][];
            empty[0] = new float[1];
            return empty;
        }

        var sut = new NeuralNetworkv7(
            _networkLayer,
            (biases) => _biases.Prepend(new float[1]).ToArray(),
            (weights) => _weights.Prepend(createEmpty()).ToArray());

        sut.StochasticGradientDescent(_data, 0.5f);

        var yPredicted0 = sut.Predict(_data[0].Features);
        var yPredicted1 = sut.Predict(_data[1].Features);
        var yPredicted2 = sut.Predict(_data[2].Features);
        var yPredicted3 = sut.Predict(_data[3].Features);
        var yPredicted4 = sut.Predict(_data[4].Features);
        var yPredicted5 = sut.Predict(_data[5].Features);
        var yPredicted6 = sut.Predict(_data[6].Features);
        var yPredicted7 = sut.Predict(_data[7].Features);

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
            predicted.Should().BeApproximately(expected, 0.1f);
        }
    }

    public class ModelData : ILabeledData
    {
        public float[] Features { get; set; } = [];

        public float[] Label { get; set; } = [];
    }
}