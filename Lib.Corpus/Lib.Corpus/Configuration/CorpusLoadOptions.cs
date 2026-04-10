namespace Lib.Corpus.Configuration;

public record CorpusLoadOptions(bool Lowercase = false, double ValidationFraction = 0.1, string FallbackText = "");