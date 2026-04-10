public class ReplOptions
{
    public float Temperature { get; set; }
    public int TopK { get; set; }
    public int? Seed { get; set; }
    public bool IsRunning { get; set; }
    public int MaxTokens { get; set; }

    public ReplOptions()
    {
        Temperature = 1.0f;
        TopK = 5;
        Seed = null;
        IsRunning = true;
        MaxTokens = 50;
    }
}