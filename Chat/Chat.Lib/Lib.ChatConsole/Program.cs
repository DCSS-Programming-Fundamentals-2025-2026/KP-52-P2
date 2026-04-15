using Contracts;
using Lib.Sampling;
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Консольне тестування системи Mini-ChatGPT.");
        ISampler realSampler = new Sampler();
        ITextGenerator mockModel = new IntegratedMockModel(realSampler);
        ChatRepl chat = new ChatRepl(mockModel);
        chat.Run(1.0f, 10, null);
    }
}