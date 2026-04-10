using System;
using System.IO;
using NUnit.Framework;

namespace Lib.ChatConsole.Tests
{
    public class Tests
    {
        private TextWriter _originalOut;

        [SetUp]
        public void SetUp()
        {
            _originalOut = Console.Out;
        }

        [TearDown]
        public void TearDown()
        {
            Console.SetOut(_originalOut);
        }

        private CommandExecutionContext CreateContext()
        {
            ReplOptions options = new ReplOptions();
            FakeTextGenerator generator = new FakeTextGenerator();
            return new CommandExecutionContext(options, generator);
        }

        [Test]
        public void TempCommand_Execute_DoesNotThrow()
        {
            TempCommand command = new TempCommand();
            CommandExecutionContext context = CreateContext();

            Assert.DoesNotThrow(() =>
            {
                command.Execute(new string[] { "/temp", "0.5" }, context);
            });
        }

        [Test]
        public void TempCommand_InvalidValue_DoesNotChangeTemperature()
        {
            TempCommand command = new TempCommand();
            CommandExecutionContext context = CreateContext();
            float before = context.Options.Temperature;

            command.Execute(new string[] { "/temp", "abc" }, context);

            Assert.That(context.Options.Temperature, Is.EqualTo(before));
        }

        [Test]
        public void TopKCommand_ValidValue_ChangesTopK()
        {
            TopKCommand command = new TopKCommand();
            CommandExecutionContext context = CreateContext();

            command.Execute(new string[] { "/topk", "7" }, context);

            Assert.That(context.Options.TopK, Is.EqualTo(7));
        }

        [Test]
        public void TopKCommand_Zero_DoesNotChangeTopK()
        {
            TopKCommand command = new TopKCommand();
            CommandExecutionContext context = CreateContext();
            int before = context.Options.TopK;

            command.Execute(new string[] { "/topk", "0" }, context);

            Assert.That(context.Options.TopK, Is.EqualTo(before));
        }

        [Test]
        public void SeedCommand_ValidInteger_ChangesSeed()
        {
            SeedCommand command = new SeedCommand();
            CommandExecutionContext context = CreateContext();

            command.Execute(new string[] { "/seed", "42" }, context);

            Assert.That(context.Options.Seed, Is.EqualTo(42));
        }

        [Test]
        public void SeedCommand_Null_SetsSeedToNull()
        {
            SeedCommand command = new SeedCommand();
            CommandExecutionContext context = CreateContext();
            context.Options.Seed = 99;

            command.Execute(new string[] { "/seed", "null" }, context);

            Assert.That(context.Options.Seed, Is.Null);
        }

        [Test]
        public void QuitCommand_Execute_StopsRepl()
        {
            QuitCommand command = new QuitCommand();
            CommandExecutionContext context = CreateContext();
            context.Options.IsRunning = true;

            command.Execute(new string[] { "/quit" }, context);

            Assert.That(context.Options.IsRunning, Is.False);
        }

        [Test]
        public void HelpCommand_Execute_PrintsRegisteredCommands()
        {
            CommandRegistry registry = new CommandRegistry();
            registry.Register(new TempCommand());
            registry.Register(new QuitCommand());

            HelpCommand command = new HelpCommand(registry);
            CommandExecutionContext context = CreateContext();

            StringWriter writer = new StringWriter();
            Console.SetOut(writer);

            command.Execute(new string[] { "/help" }, context);

            string output = writer.ToString();

            StringAssert.Contains("/temp", output);
            StringAssert.Contains("/quit", output);
        }

        private class FakeTextGenerator : ITextGenerator
        {
            public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
            {
                return "fake";
            }
        }

        [Test]
        public void ResetCommand_RestoresDefaultOptions()
        {
            ReplOptions options = new ReplOptions();
            options.Temperature = 0.5f;
            options.TopK = 5;
            options.Seed = 42;
            options.MaxTokens = 100;

            BasicModel generator = new BasicModel();
            CommandExecutionContext context = new CommandExecutionContext(options, generator);
            ResetCommand command = new ResetCommand();
            string[] args = new string[] { "/reset" };

            command.Execute(args, context);

            Assert.That(options.Temperature, Is.EqualTo(1.0f));
            Assert.That(options.TopK, Is.EqualTo(5));
            Assert.That(options.Seed, Is.Null);
            Assert.That(options.MaxTokens, Is.EqualTo(50));
        }

        [Test]
        public void CommandRegistry_GetCommand_ReturnsCorrectCommand()
        {
            CommandRegistry registry = new CommandRegistry();
            TempCommand tempCmd = new TempCommand();
            registry.Register(tempCmd);

            IReplCommand foundCmd = registry.GetCommand("/temp");

            Assert.That(foundCmd, Is.Not.Null);
            Assert.That(foundCmd.Name, Is.EqualTo("/temp"));
        }

        [Test]
        public void CommandRegistry_GetCommand_ReturnsNull()
        {
            CommandRegistry registry = new CommandRegistry();

            IReplCommand foundCmd = registry.GetCommand("/ololololo");

            Assert.That(foundCmd, Is.Null, "Реєстр має повертати null для неіснуючих команд!");
        }

        [Test]
        public void MaxTokensCommand_UpdatesMaxTokens()
        {
            ReplOptions options = new ReplOptions();
            options.MaxTokens = 50; 
            
            BasicModel generator = new BasicModel();
            CommandExecutionContext context = new CommandExecutionContext(options, generator);
            
            MaxTokensCommand command = new MaxTokensCommand();
            string[] args = new string[] { "/maxtokens", "200" };

            command.Execute(args, context);

            Assert.That(options.MaxTokens, Is.EqualTo(200));
        }

        [Test]
        public void ChatRepl_GeneratesText_WhenPromptEntered()
        {
            BasicModel generator = new BasicModel();
            ChatRepl chat = new ChatRepl(generator);
            
            StringReader simulatedInput = new StringReader("тест\n/quit\n");
            StringWriter simulatedOutput = new StringWriter();
            
            Console.SetIn(simulatedInput);
            Console.SetOut(simulatedOutput);

            chat.Run(1.0f, 5, null);

            string consoleOutput = simulatedOutput.ToString();
            
            Assert.That(consoleOutput, Does.Contain("Модель>"));

            StreamReader standardIn = new StreamReader(Console.OpenStandardInput());
            StreamWriter standardOut = new StreamWriter(Console.OpenStandardOutput());
            Console.SetIn(standardIn);
            Console.SetOut(standardOut);
        }
    }
}