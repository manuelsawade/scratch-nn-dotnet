using Microsoft.Extensions.Configuration;
using ScratchNN.App.DataTransformations;

namespace ScratchNN.App;

internal class DataPreparation
{
    internal static (LabeledData[], LabeledData[]) Prepare(IConfigurationRoot config)
    {
        var trainingSamples = DataReader
            .ReadFile(config["Paths:DataPath"]!, config["Paths:TrainingFile"]!)
            .ToArray();

        var testSamples = DataReader
            .ReadFile(config["Paths:DataPath"]!, config["Paths:TestFile"]!)
            .ToArray();

        var trainingData = PrepareData(trainingSamples);
        var testData = PrepareData(testSamples);

        return (trainingData, testData);
    }

    private static LabeledData[] PrepareData(SampleData[] samples)
    {
        var allLabels = samples.Select(data => data.Label).ToArray();
        var allFeatures = samples.Select(data => data.InputData).ToArray();

        var encodedLabels = OneHotEncoding.Transform(allLabels);
        var standardizedFeatures = Standardizer.Transform(allFeatures);

        var trainingData = Enumerable
            .Zip(encodedLabels, standardizedFeatures)
            .Select((sample) => new LabeledData
            {
                ExpectedData = sample.First,
                InputData = sample.Second,
            })
            .ToArray();

        return trainingData;
    }
}
