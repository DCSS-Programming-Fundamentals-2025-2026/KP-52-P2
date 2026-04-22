using Contracts;

public class CommandExecutionContext
{
    private ReplOptions _options;
    private ITextGenerator _generator;

    public ReplOptions Options
    {
        get { return _options; }
    }

    public ITextGenerator Generator
    {
        get { return _generator; }
    }

    public CommandExecutionContext(ReplOptions options, ITextGenerator generator)
    {
        _options = options;
        _generator = generator;
    }
}