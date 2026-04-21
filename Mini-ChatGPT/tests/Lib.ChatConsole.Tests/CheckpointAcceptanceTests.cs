using System;
using System.IO;
using System.Text.Json;
using Contracts;
using NUnit.Framework;

namespace Lib.ChatConsole.Tests
{
    [TestFixture]
    public class CheckpointAcceptanceTests
    {
        private string _tempDirectory = string.Empty;

        [SetUp]
        public void SetUp()
        {
            _tempDirectory = Path.Combine(
                Path.GetTempPath(),
                "KP52P2_CheckpointTests_",
                Guid.NewGuid().ToString("N"));

            Directory.CreateDirectory(_tempDirectory);
        }

        [TearDown]
        public void TearDown()
        {
            if (!string.IsNullOrWhiteSpace(_tempDirectory) && Directory.Exists(_tempDirectory))
            {
                Directory.Delete(_tempDirectory, true);
            }
        }

        [Test]
        public void Save_Then_Load_RestoresRequiredCheckpointFields()
        {
            string path = Path.Combine(_tempDirectory, "checkpoint.json");

            Checkpoint checkpoint = new Checkpoint(
                "trigram",
                "word",
                new
                {
                    language = "uk",
                    words = new string[] { "Привіт", "світ" }
                },
                new
                {
                    state = "trained",
                    vocabSize = 2
                },
                42,
                "V1_trigram:vocabSize=2");

            JsonCheckpointIO.Save(path, checkpoint);

            Checkpoint loaded = JsonCheckpointIO.Load(path);

            Assert.That(loaded.ModelKind, Is.EqualTo("trigram"));
            Assert.That(loaded.TokenizerKind, Is.EqualTo("word"));
            Assert.That(loaded.Seed, Is.EqualTo(42));
            Assert.That(loaded.ContractFingerprintChain, Is.EqualTo("V1_trigram:vocabSize=2"));
            Assert.That(loaded.TokenizerPayload, Is.Not.Null);
            Assert.That(loaded.ModelPayload, Is.Not.Null);

            JsonElement tokenizerPayload = (JsonElement)loaded.TokenizerPayload;
            JsonElement modelPayload = (JsonElement)loaded.ModelPayload;

            Assert.That(tokenizerPayload.ValueKind, Is.EqualTo(JsonValueKind.Object));
            Assert.That(modelPayload.ValueKind, Is.EqualTo(JsonValueKind.Object));

            Assert.That(tokenizerPayload.GetProperty("language").GetString(), Is.EqualTo("uk"));

            JsonElement words = tokenizerPayload.GetProperty("words");
            Assert.That(words.ValueKind, Is.EqualTo(JsonValueKind.Array));
            Assert.That(words.GetArrayLength(), Is.EqualTo(2));
            Assert.That(words[0].GetString(), Is.EqualTo("Привіт"));
            Assert.That(words[1].GetString(), Is.EqualTo("світ"));

            Assert.That(modelPayload.GetProperty("state").GetString(), Is.EqualTo("trained"));
            Assert.That(modelPayload.GetProperty("vocabSize").GetInt32(), Is.EqualTo(2));
        }

        [Test]
        public void Save_Then_Load_Preserves_Ukrainian_Text_InPayload()
        {
            string path = Path.Combine(_tempDirectory, "checkpoint_uk.json");

            Checkpoint checkpoint = new Checkpoint(
                "bigram",
                "word",
                new
                {
                    corpus = "Привіт, Україно!",
                    note = "тест української мови"
                },
                new
                {
                    modelName = "перевірка"
                },
                777,
                "V1_bigram:vocabSize=10");

            JsonCheckpointIO.Save(path, checkpoint);

            Checkpoint loaded = JsonCheckpointIO.Load(path);

            JsonElement tokenizerPayload = (JsonElement)loaded.TokenizerPayload;
            JsonElement modelPayload = (JsonElement)loaded.ModelPayload;

            Assert.That(tokenizerPayload.GetProperty("corpus").GetString(), Is.EqualTo("Привіт, Україно!"));
            Assert.That(tokenizerPayload.GetProperty("note").GetString(), Is.EqualTo("тест української мови"));
            Assert.That(modelPayload.GetProperty("modelName").GetString(), Is.EqualTo("перевірка"));
        }
    }
}