namespace Lib.Corpus.Processing;

public static class CorpusSplitter
{
    public static Corpus Split(string text, double validationFraction)
    {
        if (!(validationFraction >= 0.0 && validationFraction <= 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(validationFraction), "Validation fraction must be a valid number between 0.0 and 1.0.");
        }

        if (string.IsNullOrEmpty(text))
        {
            return new Corpus(string.Empty, string.Empty);
        }

        int valLength = (int)(text.Length * validationFraction);
        int trainLength = text.Length - valLength;

        string trainText = text.Substring(0, trainLength);
        string valText = text.Substring(trainLength);

        return new Corpus(trainText, valText);
    }
}