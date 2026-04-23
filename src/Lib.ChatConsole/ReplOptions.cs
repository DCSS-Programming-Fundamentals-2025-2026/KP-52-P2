public class ReplOptions
{
    public string CheckpointPath { get; set; }
    public float Temperature { get; set; } = 0.3f;
    public int TopK { get; set; } = 5;
    public int? Seed { get; set; } = 42;
    public bool IsRunning { get; set; } = true;
    public int MaxTokens { get; set; } = 50;


    public string DataPath { get; set; } = "../data/sample.txt";
    public string ModelKind { get; set; } = "trigram";
    public string TokenizerKind { get; set; } = "word";
    public int Epochs { get; set; } = 3;
    public float LearningRate { get; set; } = 0.1f;
    public int BatchSize { get; set; } = 1;
    public int BlockSize { get; set; } = 1;
    public int CheckpointInterval {get; set;} = 10;
    public string OutputPath { get; set; } = "checkpoint.json";
}