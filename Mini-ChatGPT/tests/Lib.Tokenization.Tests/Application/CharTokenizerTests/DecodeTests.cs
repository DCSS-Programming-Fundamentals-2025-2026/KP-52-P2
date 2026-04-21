using NUnit.Framework;
using Lib.Tokenization.Application;
using System;

namespace Lib.Tokenization.Tests.Application.CharTokenizerTests
{
    [TestFixture]
    public class DecodeTests
    {
        [Test]
        public void Decode_EmptyArray_ReturnsEmptyString()
        {
            var tokenizer = CharTokenizer.BuildFromText("Тест");
            int[] emptyTokens = Array.Empty<int>();

            string result = tokenizer.Decode(emptyTokens);

            Assert.That(result, Is.EqualTo(string.Empty));
        }

        [Test]
        public void Decode_WithUnkId_ReturnsDefaultChar()
        {
            var tokenizer = CharTokenizer.BuildFromText("АБ");
            int[] tokensWithUnk = new int[] { 1, 0, 2 };

            string result = tokenizer.Decode(tokensWithUnk);

            Assert.That(result[1], Is.EqualTo('\0'));
        }
    }
}