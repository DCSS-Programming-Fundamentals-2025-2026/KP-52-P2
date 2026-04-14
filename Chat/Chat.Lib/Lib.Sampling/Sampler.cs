using scr.Lib.Sampling.Processing;
using scr.Processing;

namespace scr;

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

        float[] logits = TemperatureScaler.Scale(probs, temperature);

        int[] topKIndices = TopKSelector.GetTopKIndices(logits, topK);

        float[] topKLogits = new float[topKIndices.Length];

        for (int i = 0; i < topKIndices.Length; i++)
        {
            topKLogits[i] = logits[topKIndices[i]];
        }

        float[] topKProbs = ProbabilityNormalizer.Normalize(topKLogits);

        float r = (float)rng.NextDouble();
        float cumulative = 0f;

        for (int i = 0; i < topKProbs.Length; i++)
        {
            cumulative += topKProbs[i];
            if (r <= cumulative)
            {
                return topKIndices[i];
            }
        }

        return topKIndices[^1];
    }

    public int SampleWithSeed(float[] probs, float temperature, int topK, int seed)
    {
        return Sample(probs, temperature, topK, new Random(seed));
    }
}