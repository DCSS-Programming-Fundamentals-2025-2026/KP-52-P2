using System;

public class QuitCommand : IReplCommand
{
    public string Name
    {
        get { return "/quit"; }
    }

    public string Description
    {
        get { return "Вихід із чату"; }
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        context.Options.IsRunning = false;
        Console.WriteLine("Завершення роботи...");
    }
}