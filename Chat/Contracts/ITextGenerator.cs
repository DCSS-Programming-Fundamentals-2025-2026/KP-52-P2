namespace Contracts
{
    public interface ITextGenerator
    {
        
        string Generate(string prompt, float temperature, int topK, int? seed);
    }
}