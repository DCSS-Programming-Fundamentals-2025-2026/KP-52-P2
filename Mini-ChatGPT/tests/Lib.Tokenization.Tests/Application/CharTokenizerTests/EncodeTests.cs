using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.CharTokenizerTests
{
    [TestFixture]
    public class EncodeTests
    {
        [TestCase("")]
        public void Encode_EmptyOrNullText_ReturnsEmptyArray(string inputText)
        {
            var tokenizer = CharTokenizer.BuildFromText("Тест");

            int[] result = tokenizer.Encode(inputText);

            Assert.That(result, Is.Empty);
        }

        [TestCase("АВБ", 1, 0)]
        [TestCase("АБ!", 2, 0)]
        public void Encode_WithUnknownCharacters_AssignsUnkIdZero(string textToEncode, int unknownCharIndex, int expectedId)
        {
            var tokenizer = CharTokenizer.BuildFromText("АБ");

            int[] tokens = tokenizer.Encode(textToEncode);

            Assert.That(tokens[unknownCharIndex], Is.EqualTo(expectedId));
        }
    }
}