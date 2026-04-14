namespace scr.Lib.Sampling.Processing;

public static class TopKSelector
{
    public static int[] GetTopKIndices(float[] values, int k)
    {
        if (values == null || values.Length == 0)
            throw new ArgumentException("Values cannot be null or empty.");

        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k));

        int n = values.Length;
        int actualK = Math.Min(k, n);

        int[] indices = new int[n];
        float[] copy = new float[n];

        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
            copy[i] = values[i];
        }

        Array.Sort(copy, indices);

        int[] result = new int[actualK];

        for (int i = 0; i < actualK; i++)
            result[i] = indices[n - 1 - i];

        return result;
    }
}