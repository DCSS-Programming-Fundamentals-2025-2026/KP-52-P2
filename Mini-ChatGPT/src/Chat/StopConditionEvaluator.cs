namespace Chat.TextGeneration;

public static class StopConditionEvaluator
{
    public static bool ShouldStop(string decodedToken, int generatedCount)
    {
        if (generatedCount < 5 || string.IsNullOrEmpty(decodedToken))
        {
            return false;
        }

        char[] stopCharacters = { '.', '!', '?' };
        return decodedToken.IndexOfAny(stopCharacters) >= 0;
    }
}