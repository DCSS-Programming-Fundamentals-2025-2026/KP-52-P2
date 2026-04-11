using MiniChatGPT.Contracts;

namespace Lib.Tokenization.Application
{
    public interface ITokenizerFactory
    {
        ITokenizer BuildFromText(string text);
        ITokenizer FromPayload(object payload);
    }
}