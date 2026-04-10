public class CommandRegistry
{
    private Dictionary<string, IReplCommand> _commands;

    public CommandRegistry()
    {
        _commands = new Dictionary<string, IReplCommand>();
    }

    public void Register(IReplCommand command)
    {
        if (!_commands.ContainsKey(command.Name))
        {
            _commands.Add(command.Name, command);
        }
    }

    public IReplCommand GetCommand(string name)
    {
        if (_commands.ContainsKey(name))
        {
            return _commands[name];
        }
        return null;
    }

    public List<IReplCommand> GetAllCommands()
    {
        List<IReplCommand> list = new List<IReplCommand>();
        
        foreach (KeyValuePair<string, IReplCommand> kvp in _commands)
        {
            list.Add(kvp.Value);
        }
        return list;
    }
}