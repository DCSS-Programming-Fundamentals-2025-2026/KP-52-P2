public interface IReplCommand
{
    string Name { get; }
    string Description { get; }

    void Execute(string[] args, CommandExecutionContext context);
}