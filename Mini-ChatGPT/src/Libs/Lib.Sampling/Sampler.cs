using Lib.Sampling.Processing;
using Lib.MathCore;
namespace Lib.Sampling;

public class Sampler : ISampler
{
    public int Sample(float[] probs, float temperature, int topK, Random? rng)
    {
        if (probs == null || probs.Length == 0)
        {
            throw new ArgumentException("Probs cannot be null or empty.");
        }

        if (rng == null)
        {
            rng = new Random();
        }

        if (temperature < 0.05f || topK == 1)
        {
            return MathOps.Default.ArgMax(probs);
        }

        float[] logits = TemperatureScaler.Scale(probs, temperature);

        int[] topKIndices = TopKSelector.GetTopKIndices(logits, topK);

        float[] topKLogits = new float[topKIndices.Length];

        for (int i = 0; i < topKIndices.Length; i++)
        {
            topKLogits[i] = logits[topKIndices[i]];
        }

        float[] topKProbs = ProbabilityNormalizer.Normalize(topKLogits);

        int localIndex = MathOps.Default.SampleFromProbs(topKProbs, rng);

        return topKIndices[localIndex];
    }

    public int SampleWithSeed(float[] probs, float temperature, int topK, int seed)
    {
        return Sample(probs, temperature, topK, new Random(seed));
    }
}