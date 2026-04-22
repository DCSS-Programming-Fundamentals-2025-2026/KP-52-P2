using System;
using System.Collections.Generic;

public class HelpCommand : IReplCommand
{
    private CommandRegistry _registry;

    public HelpCommand(CommandRegistry registry)
    {
        _registry = registry;
    }

    public string Name
    {
        get { return "/help"; }
    }

    public string Description
    {
        get { return "Показує список доступних команд"; }
    }

    public void Execute(string[] args, CommandExecutionContext context)
    {
        List<IReplCommand> commands = _registry.GetAllCommands();

        Console.WriteLine("Доступні команди:");

        for (int i = 0; i < commands.Count; i++)
        {
            Console.WriteLine(commands[i].Name + " - " + commands[i].Description);
        }
    }
}
