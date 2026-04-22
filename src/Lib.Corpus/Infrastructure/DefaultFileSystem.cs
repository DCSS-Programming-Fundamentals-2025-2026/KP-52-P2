namespace Lib.Corpus.Infrastructure;

public class DefaultFileSystem : IFileSystem
{
    public bool Exists(string path) => File.Exists(path);

    public string ReadAllText(string path) => File.ReadAllText(path);
}