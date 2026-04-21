using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.CharTokenizerTests
{
    [TestFixture]
    public class RoundTripTests
    {
        [TestCase("Привіт, світе! 123")]
        [TestCase("AaBbCc")]
        [TestCase("               ")]
        [TestCase("!?.,:;\"'")]
        public void EncodeAndDecode_ValidText_ReturnsExactOriginalText(string inputText)
        {
            var tokenizer = CharTokenizer.BuildFromText(inputText);

            int[] encodedTokens = tokenizer.Encode(inputText);
            string decodedText = tokenizer.Decode(encodedTokens);

            Assert.That(decodedText, Is.EqualTo(inputText));
        }
    }
}