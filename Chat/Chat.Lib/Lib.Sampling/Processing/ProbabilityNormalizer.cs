namespace scr.Processing;

public static class ProbabilityNormalizer
{
    public static float[] Normalize(float[] logits)
    {
        if (logits == null)
            throw new ArgumentNullException(nameof(logits));

        if (logits.Length == 0)
            throw new ArgumentException("Logits cannot be empty.", nameof(logits));

        float[] probs = new float[logits.Length];

        float maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > maxLogit)
                maxLogit = logits[i];

        float sumExp = 0f;

        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = (float)Math.Exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }

        if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            float uniform = 1f / probs.Length;
            for (int i = 0; i < probs.Length; i++)
                probs[i] = uniform;

            return probs;
        }

        for (int i = 0; i < probs.Length; i++)
            probs[i] /= sumExp;

        return probs;
    }
}