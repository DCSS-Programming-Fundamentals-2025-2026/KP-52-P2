using scr;

public class IntegratedMockModel : ITextGenerator
{
    private ISampler _sampler;
    
    private string[] _vocab = new string[] { "фора", "компот", "бот", "лололо", "гроші", "потужно" };
    
    private float[] _mockLogits = new float[] { 2.0f, 1.5f, 5.0f, 0.1f, 3.0f, 4.0f };

    public IntegratedMockModel(ISampler sampler)
    {
        _sampler = sampler;
    }

    public string Generate(string prompt, int maxTokens, float temperature, int topk, int? seed)
    {
        int selectedIndex;

        if (seed.HasValue)
        {
            selectedIndex = _sampler.SampleWithSeed(_mockLogits, temperature, topk, seed.Value);
        }
        else
        {
            selectedIndex = _sampler.Sample(_mockLogits, temperature, topk, (Random)null);
        }

        return _vocab[selectedIndex];
    }
}