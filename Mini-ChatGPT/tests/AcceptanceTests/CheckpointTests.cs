using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Tests;
using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Tokenization.Application;
using Lib.Training;
using Lib.Training.Configuration;
using MiniChatGPT.Contracts;
using System;
using System.Collections.Generic;
using System.Text;

namespace AcceptanceTests
{
    public class CheckpointTests
    {
        [Test]
        public void CheckpointTrigram_ContainsValidFields()
        {
            // Arrange
            string tokenizerKind = "word";
            int seed = 42;

            CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load("../../../../../data/showcase2.txt", new CorpusLoadOptions { Lowercase = true });

            WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            int[] tokens = tokenizer.Encode(corpus.TrainText);

            TrigramModel model = new TrigramModel(tokenizer.VocabSize);
            model.Train(tokens);

            ITokenStream tokenStream = new ArrayTokenStream(tokens);
            BatchWindowSampler windowSampler = new BatchWindowSampler();
            IBatchProvider batchProvider = new TokenBatchProvider(tokenStream, windowSampler);

            TrainingLoop trainingLoop = new TrainingLoop();
            TrainingConfig tConfig = new TrainingConfig(100, 0.1f, 10);
            BatchConfig bConfig = new BatchConfig(50, 16);

            var metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json");

            Checkpoint checkpoint = new Checkpoint(
                ModelKind: model.ModelKind,
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed,
                ContractFingerprintChain: $"{loader.GetContractFingerprint()}|{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
            );

            JsonCheckpointIO checkpointIO = new JsonCheckpointIO();

            // Act
            checkpointIO.Save("../../../../../data/checkpoints/NGramCheckpoints.json", checkpoint);
            Checkpoint load = checkpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");

            // Assert
            Assert.That(load, Is.Not.Null);
            Assert.That(load.ModelKind, Is.EqualTo(model.ModelKind));
            Assert.That(load.TokenizerKind, Is.EqualTo("word"));
            Assert.That(load.Seed, Is.EqualTo(42));
        }
    }
}
