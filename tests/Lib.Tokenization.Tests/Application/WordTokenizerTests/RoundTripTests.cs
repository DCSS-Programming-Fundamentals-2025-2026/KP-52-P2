using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.WordTokenizerTests
{
    [TestFixture]
    public class RoundTripTests
    {
        [TestCase("простий тестовий рядок", "простий тестовий рядок")]
        [TestCase("Привіт,!", "привіт")]
        [TestCase("Слова.!. з купою: пунктуації???", "слова з купою пунктуації")]
        [TestCase("    зайві   пробіли   ", "зайві пробіли")]
        [TestCase("Хай живе батько-наш Бандера!", "хай живе батько-наш бандера")] 
        [TestCase("    текстове ...,,,!!???   ля", "текстове ля")]
        public void EncodeAndDecode_VariousTexts_ReturnsNormalizedText(string inputText, string expectedOutput)
        {
            var tokenizer = WordTokenizer.BuildFromText(inputText);

            int[] encodedTokens = tokenizer.Encode(inputText);
            string decodedText = tokenizer.Decode(encodedTokens);

            Assert.That(decodedText, Is.EqualTo(expectedOutput));
        }
    }
}