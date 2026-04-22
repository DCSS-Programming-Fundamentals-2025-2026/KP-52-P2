using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.WordTokenizerTests
{
    [TestFixture]
    public class EncodeTests
    {
        [TestCase("")]
        [TestCase("   ")]
        [TestCase(" ")]
        public void Encode_EmptyOrWhitespaceText_ReturnsEmptyArray(string inputText)
        {
            var tokenizer = WordTokenizer.BuildFromText("Тестовий словник");

            int[] tokens = tokenizer.Encode(inputText);

            Assert.That(tokens, Is.Empty);
        }

        [Test]
        public void Encode_WithUnknownWords_AssignsUnkIdZero()
        {
            var tokenizer = WordTokenizer.BuildFromText("proZZoro Kurvet");
            string textToEncode = "proZZoro frog Kurvet";

            int[] tokens = tokenizer.Encode(textToEncode);

            Assert.That(tokens.Length, Is.EqualTo(3));
            Assert.That(tokens[1], Is.EqualTo(0)); 
        }
    }
}