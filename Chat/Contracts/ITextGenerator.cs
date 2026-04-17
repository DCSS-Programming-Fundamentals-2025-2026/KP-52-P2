namespace Contracts
{
    public interface ITextGenerator
    {
        string Generate(string prompt, int maxTokens=50,
        float temperature=0.7f, int topK = 40, int? seed = null);
    }
}