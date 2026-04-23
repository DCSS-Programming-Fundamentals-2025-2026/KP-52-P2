public class ResetCommand : IReplCommand
{
    public string Name 
    { 
        get { return "/reset"; } 
    }

    public string Description 
    { 
        get { return "Скинути контекст чату (очищує консоль)"; } 
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        context.Options.Temperature = 1.0f;
        context.Options.TopK = 5;
        context.Options.Seed = null;
        context.Options.MaxTokens = 50;

        try
        {
            Console.Clear();
        }
        catch (IOException)
        {
        }
        Console.WriteLine("Контекст скинуто. Можете почати нову розмову.");
    }
}