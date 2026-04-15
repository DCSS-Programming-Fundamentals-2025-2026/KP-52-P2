public class ReplOptions
{
    public string CheckpointPath { get; set; } = "checkpoint.json";
    public float Temperature { get; set; } = 0.3f;
    public int TopK { get; set; } = 5;
    public int? Seed { get; set; } = null;
    public bool IsRunning { get; set; } = true;
    public int MaxTokens { get; set; } = 50;
}