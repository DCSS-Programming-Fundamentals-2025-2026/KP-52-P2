public class MaxTokensCommand : IReplCommand
{
    public string Name 
    { 
        get { return "/maxtokens"; } 
    }

    public string Description 
    { 
        get { return "Змінити максимальну кількість токенів (слів) у відповіді"; } 
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Помилка: вкажіть число токенів (наприклад: /maxtokens 100).");
            return;
        }

        int newMax;
        if (int.TryParse(args[1], out newMax))
        {
            if (newMax > 0)
            {
                context.Options.MaxTokens = newMax;
                Console.WriteLine("Ліміт токенів успішно змінено на " + newMax);
            }
            else
            {
                Console.WriteLine("Помилка: ліміт токенів має бути більшим за нуль.");
            }
        }
        else
        {
            Console.WriteLine("Помилка: некоректний формат цілого числа.");
        }
    }
}