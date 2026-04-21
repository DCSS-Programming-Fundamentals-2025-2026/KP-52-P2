using System.Text.Json;
using Contracts;
using Lib.Tokenization.Application;

namespace Lib.Tokenization.Infrastructure.Serialization
{
    public static class TokenizerPayloadSerializer
    {
        public static ITokenizer RestoreTokenizer(string tokenizerKind, JsonElement payload)
        {
            if (tokenizerKind.ToLower() == "char")
            {
                return new CharTokenizerFactory().FromPayload(payload);
            }
            else if (tokenizerKind.ToLower() == "word")
            {
                return new WordTokenizerFactory().FromPayload(payload);
            }
            
            throw new ArgumentException($"Unknown tokenizer type: {tokenizerKind}");
        }
    }
}