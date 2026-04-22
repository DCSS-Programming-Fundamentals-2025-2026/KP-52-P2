public class TempCommand : IReplCommand
{
    public string Name 
    { 
        get { return "/temp"; } 
    }

    public string Description 
    { 
        get { return "Змінити температуру (наприклад: /temp 0.5)"; } 
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Помилка: вкажіть значення температури.");
            return;
        }

        float newTemp;
        if (float.TryParse(args[1], out newTemp))
        {
            context.Options.Temperature = newTemp;
            Console.WriteLine("Температуру змінено на " + newTemp);
        }
        else
        {
            Console.WriteLine("Помилка: некоректний формат числа.");
        }
    }
}