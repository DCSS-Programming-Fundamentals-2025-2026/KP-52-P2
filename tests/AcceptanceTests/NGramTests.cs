using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Streams;
using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Models.NGram;
using Lib.Models.NGram.Factories;
using Lib.Models.Trigram;
using Lib.Tokenization.Application;
using Lib.Tokenization.Infrastructure.Serialization;
using Lib.Training;
using Lib.Training.Configuration;
using System.Text.Json;

namespace AcceptanceTests
{
    public class NGramTests
    {
        private int? seed;

        [SetUp]
        public void Setup()
        {
            seed = 42;
        }

        [TestCase("../../../../../data/showcase.txt")]
        [TestCase("../../../../../data/showcase2.txt")]
        [TestCase("../../../../../data/showcase3.txt")]
        [TestCase("../../../../../data/max_trigram_data.txt")]
        public void Bigram_FullCycleTillCheckpointLoad(string dataPath)
        {
            // Arrange
            string tokenizerKind = "word";
            int seed = 42;

            CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load(dataPath, new CorpusLoadOptions { Lowercase = true });

            WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            int[] tokens = tokenizer.Encode(corpus.TrainText);

            NGramModel model = new NGramModel(tokenizer.VocabSize);
            model.Train(tokens);

            ITokenStream tokenStream = new ArrayTokenStream(tokens);
            BatchWindowSampler windowSampler = new BatchWindowSampler();
            IBatchProvider batchProvider = new TokenBatchProvider(tokenStream, windowSampler);

            TrainingLoop trainingLoop = new TrainingLoop();
            TrainingConfig tConfig = new TrainingConfig(100, 0.1f, 10);
            BatchConfig bConfig = new BatchConfig(50, 16);

            var metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", tokenizerKind, tokenizer, seed);

            Checkpoint checkpoint = new Checkpoint(
                ModelKind: model.ModelKind,
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed,
                ContractFingerprintChain: $"{loader.GetContractFingerprintChain()}|{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
            );

            // Act
            JsonCheckpointIO.Save("../../../../../data/checkpoints/NGramCheckpoints.json", checkpoint);
            Checkpoint loaded = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");

            JsonElement tokenizerPayload = (JsonElement)loaded.TokenizerPayload;
            JsonElement wordsElement = tokenizerPayload.GetProperty("Words");
            string[] words = wordsElement.Deserialize<string[]>() ?? Array.Empty<string>();
            tokenizerPayload = JsonSerializer.SerializeToElement(new { Words = words });

            var restoredTokenizer = TokenizerPayloadSerializer.RestoreTokenizer(
                loaded.TokenizerKind,
                tokenizerPayload
            );

            var factory = new NGramModelFactory();
            var restoredModel = factory.Create(loaded.ModelKind, restoredTokenizer.VocabSize);

            NGramModel newModel = (NGramModel)restoredModel;
            newModel.FromPayload((JsonElement)loaded.ModelPayload);

            // Assert
            Assert.That(newModel.Equals(model));
        }


        [TestCase("../../../../../data/short_input.txt")]
        public void Trigram_FullCycleTillCheckpointLoad(string dataPath)
        {
            // Arrange
            string tokenizerKind = "word";
            int seed = 42;

            CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load(dataPath, new CorpusLoadOptions { Lowercase = true });

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

            var metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", tokenizerKind, tokenizer, seed);

            Checkpoint checkpoint = new Checkpoint(
                ModelKind: model.ModelKind,
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed,
                ContractFingerprintChain: $"{loader.GetContractFingerprintChain()}|{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
            );

            // Act
            JsonCheckpointIO.Save("../../../../../data/checkpoints/NGramCheckpoints.json", checkpoint);
            Checkpoint loaded = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");

            JsonElement tokenizerPayload = (JsonElement)loaded.TokenizerPayload;
            JsonElement wordsElement = tokenizerPayload.GetProperty("Words");
            string[] words = wordsElement.Deserialize<string[]>() ?? Array.Empty<string>();
            tokenizerPayload = JsonSerializer.SerializeToElement(new { Words = words });

            var restoredTokenizer = TokenizerPayloadSerializer.RestoreTokenizer(
                loaded.TokenizerKind,
                tokenizerPayload
            );

            var factory = new NGramModelFactory();
            var restoredModel = factory.Create(loaded.ModelKind, restoredTokenizer.VocabSize);
            
            TrigramModel newModel = (TrigramModel)restoredModel;
            newModel.FromPayload((JsonElement)loaded.ModelPayload);

            // Assert
            Assert.That(newModel.Equals(model));
        }
    }
}
