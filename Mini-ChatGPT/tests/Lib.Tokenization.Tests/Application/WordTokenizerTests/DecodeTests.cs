using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.WordTokenizerTests
{
    [TestFixture]
    public class DecodeTests
    {
        [Test]
        public void Decode_EmptyArray_ReturnsEmptyString()
        {
            var tokenizer = WordTokenizer.BuildFromText("якийсь текст");
            int[] emptyTokens = Array.Empty<int>();

            string result = tokenizer.Decode(emptyTokens);

            Assert.That(result, Is.EqualTo(string.Empty));
        }

        [Test]
        public void Decode_WithUnkId_ReturnsUnkString()
        {
            var tokenizer = WordTokenizer.BuildFromText("один два");
            int[] tokens = new int[] { 1, 0, 2 };

            string result = tokenizer.Decode(tokens);

            Assert.That(result, Is.EqualTo("один <UNK> два"));
        }
    }
}