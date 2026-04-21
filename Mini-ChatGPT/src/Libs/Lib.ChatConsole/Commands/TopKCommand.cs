public class TopKCommand : IReplCommand
{
    public string Name 
    { 
        get { return "/topk"; } 
    }

    public string Description 
    { 
        get { return "Змінити параметр Top-K (наприклад: /topk 5)"; } 
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Помилка: вкажіть значення Top-K.");
            return;
        }

        int newTopK;
        if (int.TryParse(args[1], out newTopK))
        {
            if (newTopK > 0)
            {
                context.Options.TopK = newTopK;
                Console.WriteLine("Параметр Top-K змінено на " + newTopK);
            }
            else
            {
                Console.WriteLine("Помилка: Top-K має бути більше нуля.");
            }
        }
        else
        {
            Console.WriteLine("Помилка: некоректний формат цілого числа.");
        }
    }
}