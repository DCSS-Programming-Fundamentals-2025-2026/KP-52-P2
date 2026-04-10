namespace Lib.Corpus.Infrastructure;

public interface IFileSystem
{
    bool Exists(string path);
    string ReadAllText(string path);
}