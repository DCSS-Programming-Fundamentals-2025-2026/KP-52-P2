using Lib.Corpus.Infrastructure;

namespace Integration.DataPipeline.Mocks
{
    public class MockFileSystem : IFileSystem
    {
        private readonly Dictionary<string, string> _virtualFiles = new();

        public void AddFile(string path, string content)
        {
            _virtualFiles[path] = content;
        }
        
        public bool Exists(string path) => _virtualFiles.ContainsKey(path);

        public string ReadAllText(string path)
        {
            if (_virtualFiles.TryGetValue(path, out var content))
            {
                return content;
            }
            
            throw new FileNotFoundException($"Mock file not found: {path}");
        }
    }
}