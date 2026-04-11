using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using MiniChatGPT.Contracts;

namespace Lib.Corpus;

public class CorpusLoader : ICorpusLoader, IContractFingerprint
{
    private readonly IFileSystem _fileSystem;
    public CorpusLoader(IFileSystem fileSystem)
    {
        _fileSystem = fileSystem;
    }

    public string GetContractFingerprint()
    {
        return "Lib.Corpus:1.0.0:Corpus,CorpusLoader";
    }

    public Corpus Load(string path, CorpusLoadOptions? options = null)
    {
        options ??= new CorpusLoadOptions();
        string text;

        if (_fileSystem.Exists(path))
        {
            text = _fileSystem.ReadAllText(path);
        }

        else
        {
            text = options.FallbackText;
        }

        return LoadFromText(text, options);
    }

    public Corpus LoadFromText(string text, CorpusLoadOptions? options = null)
    {
        options ??= new CorpusLoadOptions();

        string normalizedText = CorpusTextNormalizer.Normalize(text, options.Lowercase);

        return CorpusSplitter.Split(normalizedText, options.ValidationFraction);
    }
}