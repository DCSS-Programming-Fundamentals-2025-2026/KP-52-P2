using scr;
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Консольне тестування системи Mini-ChatGPT.");
        ReplOptions options = CommandLineParser.Parse(args);
        ISampler realSampler = new Sampler();
        ITextGenerator mockModel = new IntegratedMockModel(realSampler);
        ChatRepl chat = new ChatRepl(mockModel);
        chat.Run(1.0f, 10, null);
    }
}