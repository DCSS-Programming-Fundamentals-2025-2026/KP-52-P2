namespace Lib.Sampling.Processing;

public static class TemperatureScaler
{
    public static float[] Scale(float[] probs, float temperature)
    {
        if (probs == null || probs.Length == 0)
        {
            throw new ArgumentException("Probs cannot be null or empty.");
        }

        if (temperature < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature));
        }

        if (temperature == 0)
        {
            temperature = 0.000001f;
        }

        float[] result = new float[probs.Length];

        for (int i = 0; i < probs.Length; i++)
        {
            float p = Math.Max(probs[i], 1e-7f);

            if (temperature == 1.0f)
            {
                result[i] = (float)Math.Log(p);
            }
            else
            {
                result[i] = (float)Math.Log(p) / temperature;
            }
        }

        return result;
    }
}