using Microsoft.Extensions.Configuration;
using ScratchNN.App.DataTransformations;
using System.IO.Compression;

using LabeledData = (float[] InputData, float[] Expected);

namespace ScratchNN.App;

internal class DataPreparation
{
    internal static (LabeledData[], LabeledData[]) Prepare(IConfigurationRoot config)
    {        
        var trainingSamples = DataReader
            .ReadFile(config)
            .ToArray();

        var allLabels = trainingSamples.Select(data => data.Label).ToArray();
        var allFeatures = trainingSamples.Select(data => data.Input).ToArray();

        var encodedLabels = OneHotEncoding.Transform(allLabels);
        var standardizedFeatures = Standardizer.Transform(allFeatures);

        var trainingData = Enumerable
            .Zip(encodedLabels, standardizedFeatures)
            .Select((sample) => new LabeledData
            {
                Expected = sample.First,
                InputData = sample.Second,               
            })
            .ToArray();

        var testSamples = DataReader
            .ReadFile(config)
            .ToArray();

        var testSampleLables = testSamples.Select(data => data.Label).ToArray();
        var testFeatures = trainingSamples.Select(data => data.Input).ToArray();

        var encodedTestLabels = OneHotEncoding.Transform(testSampleLables);

        var testData = Enumerable
            .Zip(encodedTestLabels, testFeatures)
            .Select(sample => new LabeledData
            {
                Expected = sample.First,
                InputData = sample.Second,
            })
            .ToArray();

        return (trainingData, testData);
    }
}
