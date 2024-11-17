namespace ScratchNN.App;

internal class DataReader
{
    private const int _label = 1;
    private const int _input = 784;

    public static IEnumerable<(float Label, float[] Input)> ReadFile(string path)
    {
        var lineArray = File.ReadAllLines(path);

        foreach (var line in lineArray)
        {
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
