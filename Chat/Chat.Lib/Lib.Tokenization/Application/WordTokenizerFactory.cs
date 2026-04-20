using System.Text.Json;
using Lib.Tokenization.Domain.Model;
using Contracts;

namespace Lib.Tokenization.Application
{
    public class WordTokenizerFactory : ITokenizerFactory
    {
        public ITokenizer BuildFromText(string text)
        {
            return WordTokenizer.BuildFromText(text);
        }

        public ITokenizer FromPayload(object payload)
        {
            JsonElement json = (JsonElement)payload;
            string[] restored = json.GetProperty("words").Deserialize<string[]>();
            var vocab = new WordVocabulary();
            
            for (int i = 1; i < restored.Length; i++)
            {
                if (restored[i] != null) vocab.Add(restored[i]);
            }
            return new WordTokenizer(vocab);
        }
    }
}