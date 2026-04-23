namespace Lib.Corpus.Processing;

public static class CorpusTextNormalizer
{
    public static string Normalize(string text, bool lowercase)
    {
        if (string.IsNullOrEmpty(text))
        {
            return text;
        }

        if (lowercase)
        {
            return text.ToLowerInvariant();
        }

        return text;
    }
}
