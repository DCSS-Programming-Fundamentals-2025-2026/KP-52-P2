using Lib.Tokenization.Application;

namespace Lib.Tokenization.Tests.Application.WordTokenizerTests
{
    [TestFixture]
    public class BuildFromTextTests
    {
        [TestCase("Привіт привіт")]
        [TestCase("Привіт, привіт!")]
        [TestCase("ПРИВІТ... привіт:")]
        public void BuildFromText_WithDifferentPunctuationAndCase_CreatesOnlyOneUniqueWord(string inputText)
        {
            var tokenizer = WordTokenizer.BuildFromText(inputText);

            Assert.That(tokenizer.VocabSize, Is.EqualTo(2));
        }

        [TestCase("")]
        [TestCase("    ")]
        [TestCase("\t\n")]
        [TestCase(" ")]
        public void BuildFromText_EmptyOrWhitespace_CreatesOnlyUnkToken(string emptyText)
        {
            var tokenizer = WordTokenizer.BuildFromText(emptyText);

            Assert.That(tokenizer.VocabSize, Is.EqualTo(1));
        }
    }
}