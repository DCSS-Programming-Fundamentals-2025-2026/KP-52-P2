using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.CharTokenizerTests
{
    [TestFixture]
    public class BuildFromTextTests
    {
        [TestCase("АААБ", 3)]
        [TestCase("Aa", 3)]
        [TestCase("a b", 4)]
        [TestCase("!!!", 2)]
        public void BuildFromText_VariousStrings_CreatesCorrectVocabSize(string inputText, int expectedSize)
        {
            var tokenizer = CharTokenizer.BuildFromText(inputText);

            Assert.That(tokenizer.VocabSize, Is.EqualTo(expectedSize));
        }

        [TestCase("")]
        public void BuildFromText_EmptyOrNullString_CreatesOnlyUnkToken(string emptyText)
        {
            var tokenizer = CharTokenizer.BuildFromText(emptyText);

            Assert.That(tokenizer.VocabSize, Is.EqualTo(1));
        }
    }
}