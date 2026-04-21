using Contracts;
public class BasicModel : ITextGenerator
{
    public string Generate(string prompt, int maxTokens, float temperature, int topk, int? seed)
    {
        return "Демо-бот відповідає на: [" + prompt + "]. Поточна температура: " + temperature;
    }
}