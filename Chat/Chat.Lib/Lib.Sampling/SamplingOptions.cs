namespace Lib.Sampling;

public class SamplingOptions
{
    public float Temperature { get; set; }
    public int TopK { get; set; }
    public int? Seed { get; set; }
}