using System;
using System.IO;
using NUnit.Framework;

namespace Lib.ChatConsole.Tests
{
    [TestFixture]
    public class UkrainianReplAcceptanceTests
    {
        private TextReader _originalInput = null!;
        private TextWriter _originalOutput = null!;

        [SetUp]
        public void SetUp()
        {
            _originalInput = Console.In;
            _originalOutput = Console.Out;
        }

        [TearDown]
        public void TearDown()
        {
            Console.SetIn(_originalInput);
            Console.SetOut(_originalOutput);
        }

        [Test]
        public void ChatRepl_AcceptsUkrainianInput_AndPrintsUkrainianInterface()
        {
            StringReader input = new StringReader("Привіт, світе!\n/quit\n");
            StringWriter output = new StringWriter();

            Console.SetIn(input);
            Console.SetOut(output);

            BasicModel generator = new BasicModel();
            ChatRepl repl = new ChatRepl(generator);

            repl.Run(1.0f, 5, 42);

            string text = output.ToString();

            Assert.That(text, Does.Contain("=== Mini-ChatGPT REPL ==="));
            Assert.That(text, Does.Contain("Введіть текст для генерації або /help для списку команд."));
            Assert.That(text, Does.Contain("Користувач>"));
            Assert.That(text, Does.Contain("Модель>"));
            Assert.That(text, Does.Contain("Привіт, світе!"));
            Assert.That(text, Does.Contain("Поточна температура"));
        }
    }
}