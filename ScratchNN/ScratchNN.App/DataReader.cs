using Microsoft.Extensions.Configuration;
using System.IO.Compression;

namespace ScratchNN.App;

internal class DataReader
{
    private const int _label = 1;
    private const int _input = 784;

    public static IEnumerable<(float Label, float[] Input)> ReadFile(IConfigurationRoot config)
    {
        using var archive = ZipFile.OpenRead(config["Paths:DataPath"]!);
        using var trainingStream = archive.GetEntry(config["Paths:TrainingFile"]!)!.Open();
        using var fileReader = new StreamReader(trainingStream);

        var lineArray = fileReader.ReadToEnd().Split(Environment.NewLine);

        foreach (var line in lineArray)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            
            var valuesPerLine = line
                .Split(",")
                .Select(float.Parse);

            yield return (
                Label: valuesPerLine.First(),
                Input: valuesPerLine.Skip(_label).Take(_input).ToArray()
                );
        }
    }
}
