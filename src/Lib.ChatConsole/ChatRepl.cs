using Contracts;

public class ChatRepl
{
    private ITextGenerator _generator;
    
    private CommandRegistry _registry;
    
    private ReplOptions _options;

    public ChatRepl(ITextGenerator generator)
    {
        _generator = generator;
        _options = new ReplOptions();
        _registry = new CommandRegistry();
        
        _registry.Register(new TempCommand());
        _registry.Register(new TopKCommand());
        _registry.Register(new SeedCommand());
        _registry.Register(new ResetCommand());
        _registry.Register(new MaxTokensCommand());
        _registry.Register(new QuitCommand());
        _registry.Register(new HelpCommand(_registry));
    }

    public void Run(float temp, int topk, int? seed)
    {
        _options.Temperature = temp;
        _options.TopK = topk;
        _options.Seed = seed;
        _options.IsRunning = true;

        CommandExecutionContext context = new CommandExecutionContext(_options, _generator);

        Console.WriteLine("=== Mini-ChatGPT REPL ===");
        Console.WriteLine("Введіть текст для генерації або /help для списку команд.");

        while (_options.IsRunning)
        {
            Console.Write("\nКористувач> ");
            
            string input = Console.ReadLine();

            if (input == null)
            {
                break;
            }

            if (string.IsNullOrWhiteSpace(input))
            {
                continue;
            }

            if (input.StartsWith("/"))
            {
                string[] parts = input.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                
                IReplCommand command = _registry.GetCommand(parts[0]);

                if (command != null)
                {
                    command.Execute(parts, context);
                }
                else
                {
                    Console.WriteLine("Невідома команда. Введіть /help.");
                }
            }
            else
            {
                Console.Write("Модель> ");
                try
                {
                    string response = _generator.Generate(input, 50, _options.Temperature, _options.TopK, _options.Seed);
                    Console.WriteLine(response);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Помилка генерації: " + ex.Message);
                }
            }
        }
    }
}