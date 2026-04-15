using Lib.Sampling.Processing;
using Lib.MathCore;
namespace Lib.Sampling;

public class NucleusSampler
{
    public int Sample(float[] probs, float temperature, float topP, Random? rng)
    {
        if (probs == null || probs.Length == 0)
        {
            throw new ArgumentException("Probs cannot be null or empty.");
        }

        if (temperature <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature));
        }

        if (topP <= 0f || topP > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(topP), "TopP must be in range (0, 1].");
        }

        if (rng == null)
        {
            rng = new Random();
        }

        if (temperature < 0.05f) 
        {
            return MathOps.Default.ArgMax(probs);
        }

        float[] logits = TemperatureScaler.Scale(probs, temperature);

        float[] normalized = ProbabilityNormalizer.Normalize(logits);

        int n = normalized.Length;
        int[] indices = new int[n];
        float[] copy = new float[n];

        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
            copy[i] = normalized[i];
        }

        Array.Sort(copy, indices);
        Array.Reverse(indices);
        Array.Reverse(copy);

        float cumulative = 0f;
        int cutoff = 0;

        for (int i = 0; i < n; i++)
        {
            cumulative += copy[i];
            cutoff++;

            if (cumulative >= topP)
            {
                break;
            }
        }

        float[] nucleus = new float[cutoff];
        int[] nucleusIndices = new int[cutoff];

        float sum = 0f;

        for (int i = 0; i < cutoff; i++)
        {
            nucleus[i] = copy[i];
            nucleusIndices[i] = indices[i];
            sum += nucleus[i];
        }

        for (int i = 0; i < cutoff; i++)
            nucleus[i] /= sum;

        int localIndex = MathOps.Default.SampleFromProbs(nucleus, rng);

        return nucleusIndices[localIndex];
    }

    public int SampleWithSeed(float[] probs, float temperature, float topP, int seed)
    {
        return Sample(probs, temperature, topP, new Random(seed));
    }
}