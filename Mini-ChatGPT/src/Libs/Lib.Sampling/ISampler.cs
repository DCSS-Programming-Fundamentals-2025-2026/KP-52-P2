namespace Lib.Sampling;

public interface ISampler
{
    int Sample(float[] probs, float temperature, int topK, Random? rng);
    int SampleWithSeed(float[] probs, float temperature, int topK, int seed);
}