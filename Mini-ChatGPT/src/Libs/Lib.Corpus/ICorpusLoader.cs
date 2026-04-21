using Lib.Corpus.Configuration;

namespace Lib.Corpus;

public interface ICorpusLoader
{
    Corpus Load(string path, CorpusLoadOptions? options = null);
    Corpus LoadFromText(string text, CorpusLoadOptions? options = null);
}