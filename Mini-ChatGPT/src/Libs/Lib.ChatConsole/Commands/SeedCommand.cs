public class SeedCommand : IReplCommand
{
    public string Name 
    { 
        get { return "/seed"; } 
    }

    public string Description 
    { 
        get { return "Встановити seed для детермінованості (наприклад: /seed 42). /seed null для скидання."; } 
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Помилка: вкажіть значення seed або 'null'.");
            return;
        }

        if (args[1].ToLower() == "null")
        {
            context.Options.Seed = null;
            Console.WriteLine("Seed скинуто (випадкова генерація).");
            return;
        }

        int newSeed;
        if (int.TryParse(args[1], out newSeed))
        {
            context.Options.Seed = newSeed;
            Console.WriteLine("Seed змінено на " + newSeed);
        }
        else
        {
            Console.WriteLine("Помилка: некоректний формат числа.");
        }
    }
}