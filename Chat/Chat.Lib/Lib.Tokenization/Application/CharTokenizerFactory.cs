using System.Text.Json;
using Lib.Tokenization.Domain.Model;
using Contracts;

namespace Lib.Tokenization.Application
{
    public class CharTokenizerFactory : ITokenizerFactory
    {
        public ITokenizer BuildFromText(string text)
        {
            return CharTokenizer.BuildFromText(text);
        }

        public ITokenizer FromPayload(object payload)
        {
            JsonElement json = (JsonElement)payload;
            char[] restored = json.GetProperty("Chars").Deserialize<char[]>();
            var vocab = new Vocabulary<char>();
            
            for (int i = 1; i < restored.Length; i++)
            {
                vocab.Add(restored[i]); 
            }
            return new CharTokenizer(vocab);
        }
    }
}