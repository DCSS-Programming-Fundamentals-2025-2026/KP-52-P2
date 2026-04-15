using NUnit.Framework;
using Lib.Sampling;
using Contracts;
using System.Linq;
using System.Collections.Generic;

namespace Lib.ChatConsole.Tests
{
    [TestFixture]
    public class DeterministicTaskTests
    {
        private Sampler _sampler;

        [SetUp]
        public void Setup()
        {
            _sampler = new Sampler();
        }

        [Test]
        public void Sampler_MustReturnSameToken_ForSameSeed()
        {
            float[] probs = { 0.05f, 0.8f, 0.15f };
            int seed = 42;

            int firstResult = _sampler.SampleWithSeed(probs, 0.7f, 2, seed);
            int secondResult = _sampler.SampleWithSeed(probs, 0.7f, 2, seed);

            Assert.That(firstResult, Is.EqualTo(secondResult), "Результати відрізняються при однаковому seed!");
        }

        [Test]
        public void CheckpointData_MustDriveDeterministicSampling()
        {
            var testCheckpoint = new Checkpoint(
                ModelKind: "trigram",
                TokenizerKind: "word",
                TokenizerPayload: new { },
                ModelPayload: new { },
                Seed: 777,
                ContractFingerprintChain: "v1.0"
            );

            float[] probs = { 0.1f, 0.1f, 0.7f, 0.1f };

            int result1 = _sampler.SampleWithSeed(probs, 0.5f, 3, testCheckpoint.Seed);
            int result2 = _sampler.SampleWithSeed(probs, 0.5f, 3, testCheckpoint.Seed);

            Assert.That(result1, Is.EqualTo(result2), "Seed з об'єкта Checkpoint не забезпечив детермінізм!");
        }
    }
}