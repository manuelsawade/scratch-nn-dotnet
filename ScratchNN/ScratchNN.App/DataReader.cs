using System.IO.Compression;

namespace ScratchNN.App;

internal class DataReader
{
    public static IEnumerable<(float[] InputData, float Label)> ReadFile(string path, string file)
    {
        using var archive = ZipFile.OpenRead(path!);
        using var fileStream = archive.GetEntry(file!)!.Open();
        using var fileReader = new StreamReader(fileStream);

        var lineArray = fileReader.ReadToEnd().Split(Environment.NewLine);

        foreach (var line in lineArray)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            
            var valuesPerLine = line
                .Split(",")
                .Select(float.Parse)
                .ToArray();

            yield return (
                InputData: valuesPerLine[1..],
                Label: valuesPerLine[0]
                );
        }
    }
}
