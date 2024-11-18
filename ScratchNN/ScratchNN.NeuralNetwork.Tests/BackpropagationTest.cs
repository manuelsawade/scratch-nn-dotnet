using FluentAssertions;
using ScratchNN.NeuralNetwork;

namespace ScratchNN.NeuralNetwork.Tests;

public class BackpropagationTest
{
    private int[] _networkLayer;

    private float[] _input;
    private float[] _label;

    private float[][] _biases;
    private float[][][] _weights;

    private float[][] _expectedBiases;
    private float[][][] _expectedWeights;
   
    [SetUp]
    public void Setup()
    {
        _networkLayer = [4, 5, 3];

        _input = [7, 5, 3, 4];
        _label = [5, 6, 9];

        _biases =
        [
            [],
            [0.5f , 0.25f,    -0.2f,  0.3f, -0.5f],
            [0.05f, -0.0325f, 0.35f],
        ];

        _weights =
        [
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

        _expectedBiases =
        [
            [-0.0197303779f, 0.0252755173f, -0.0045234235f, 0.04611452f, -0.0347640328f],
            [-1.05086958f, -1.23805928f, -1.94199872f]
        ];

        _expectedWeights =
        [
            [
                [-0.138112649f, -0.098651886f, -0.0591911338f, -0.07892151f],
                [0.176928625f, 0.126377583f, 0.0758265555f, 0.101102069f],
                [-0.0316639654f, -0.0226171166f, -0.0135702705f, -0.018093694f],
                [0.32280165f, 0.2305726f, 0.138343558f, 0.184458077f],
                [-0.243348226f, -0.173820168f, -0.104292095f, -0.139056131f]
            ],
            [
                [-0.901495337f, -0.1435613f, -0.9470549f, -0.8979235f, -0.364449978f],
                [-1.06207716f, -0.169133648f, -1.11575234f, -1.0578692f, -0.429368854f],
                [-1.66595626f, -0.265300155f, -1.75015008f, -1.65935564f, -0.673500657f]
            ]
        ];
    }

    [Test]
    public void SimpleNeuralNetwork_should_calculate_backpropagation()
    {
        var sut = new SimpleNeuralNetwork(
            _networkLayer,
            _biases,
            _weights);

        var (gradientBiases, gradientWeights) = sut.Backpropagation(
            _input,
            _label);

        AssertBiases(
            gradientBiases,
            _expectedBiases);

        AssertWeights(
            gradientWeights,
            _expectedWeights);
    }

    private static void AssertBiases<TType>(
        TType[][] calculatedLayers, 
        TType[][] expectedLayers)
    {
        foreach (var (calculatedNeurons, expectedNeurons) in calculatedLayers.Zip(expectedLayers))
            foreach (var (calculatedBias, expectedBias) in calculatedNeurons.Zip(expectedNeurons))
            {
                calculatedBias.Should().Be(expectedBias);
            }
    }

    private static void AssertApproximationBiases(
        float[][] calculatedLayers,
        float[][] expectedLayers)
    {
        foreach (var (calculatedNeurons, expectedNeurons) in calculatedLayers.Zip(expectedLayers))
            foreach (var (calculatedBias, expectedBias) in calculatedNeurons.Zip(expectedNeurons))
            {
                calculatedBias.Should().BeApproximately(expectedBias, 0.0000001f);
            }
    }

    private static void AssertWeights<T>(
        T[][][] calculatedLayers,
        T[][][] expectedLayers)
    {
        foreach (var (calculatedNeurons, expectedNeurons) in calculatedLayers.Zip(expectedLayers))
            foreach (var (calculatedWeights, expectedWeights) in calculatedNeurons.Zip(expectedNeurons))
                foreach (var (calculatedWeight, expectedWeight) in calculatedWeights.Zip(expectedWeights))
                {
                    calculatedWeight.Should().Be(expectedWeight);
                }
    }

    private static void AssertApproximationWeights(
        float[][][] calculatedLayers,
        float[][][] expectedLayers)
    {
        foreach (var (calculatedNeurons, expectedNeurons) in calculatedLayers.Zip(expectedLayers))
            foreach (var (calculatedWeights, expectedWeights) in calculatedNeurons.Zip(expectedNeurons))
                foreach (var (calculatedWeight, expectedWeight) in calculatedWeights.Zip(expectedWeights))
                {
                    calculatedWeight.Should().BeApproximately(expectedWeight, 0.0000001f);
                }
    }
}