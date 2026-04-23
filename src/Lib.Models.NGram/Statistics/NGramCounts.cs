public class NGramCounts
{
    private const int UnknownTokenId = 0;
    public void CountBigrams(float[][] probs, ReadOnlySpan<int> tokens)
    {
        if (probs == null || probs.Length == 0)
        {
            throw new ArgumentException("Probability matrix cannot be empty.", nameof(probs));
        }

        for (int i = 1; i < tokens.Length; i++)
        {
            int prev = tokens[i - 1];
            int next = tokens[i];

            if (prev == UnknownTokenId || next == UnknownTokenId)
            {
                continue;
            }
            if (prev < 0 || prev >= probs.Length ||
                next < 0 || next >= probs.Length)
            {
                    throw new ArgumentOutOfRangeException(nameof(tokens), "Token out of vocabulary range.");
            }

            probs[prev][next]++;
        }
    }

    public void CountTrigrams(Dictionary<(int, int), float[]> probs, ReadOnlySpan<int> tokens)
    {
        if (probs == null || probs.Count == 0)
        {
            throw new ArgumentException("Probability dictionary cannot be empty.", nameof(probs));
        }

        int vocabSize = GetVocabularySize(probs);

        
        for (int i = 2; i < tokens.Length; i++)
        {
            int prev2 = tokens[i - 2];
            int prev1 = tokens[i - 1];
            int next = tokens[i];

            if (prev2 == UnknownTokenId || prev1 == UnknownTokenId || next == UnknownTokenId)
            {
                continue;
            }

            if (prev2 < 0 || prev2 >= vocabSize ||
                prev1 < 0 || prev1 >= vocabSize ||
                next < 0 || next >= vocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), "Token out of vocabulary range.");
            }

            probs[(prev2, prev1)][next]++;
        }
    }

    public int GetBigramPrevTotal(float[][] probs, int vocabSize, int prev)
    {
        int sum = 0;

        for (int i = 0; i < vocabSize; i++)
        {
            sum += (int)probs[prev][i];
        }

        return sum;
    }

    public int GetTrigramPrevsTotal(Dictionary<(int, int), float[]> probs, (int, int) pair)
    {
        int sum = 0;

        for (int i = 0; i < probs[pair].Length; i++)
        {
            sum += (int)probs[pair][i];
        }

        return sum;
    }

    private static int GetVocabularySize(Dictionary<(int, int), float[]> probs)
    {
        using var enumerator = probs.Values.GetEnumerator();

        if (!enumerator.MoveNext() || enumerator.Current == null)
        {
            throw new InvalidOperationException("Probability dictionary is empty.");
        }

        return enumerator.Current.Length;
    }
}