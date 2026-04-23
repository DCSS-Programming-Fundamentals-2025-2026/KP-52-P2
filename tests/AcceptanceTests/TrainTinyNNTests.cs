using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Streams;
using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.State;
using Lib.Tokenization.Application;
using Lib.Training;
using Lib.Training.Configuration;
using System.Text.Json;

namespace AcceptanceTests
{
    public class TrainTinyNNTests
    {
        private int? seed;
        private TinyNNModelFactory _tinyNNFactory;
        private BatchConfig _batchConfig;
        private TrainingConfig _trainingConfig;
        private TrainingLoop _training;
        private CorpusLoader _loader;
        private CorpusLoadOptions _loadOptions;
        private string _modelKind;

        [SetUp]
        public void Setup()
        {
            seed = 42;
            _batchConfig = new BatchConfig(8, 16);
            _training = new TrainingLoop();
            _trainingConfig = new TrainingConfig(10, 0.01f, 2);
            _loader = new CorpusLoader(new DefaultFileSystem());
            _loadOptions = new CorpusLoadOptions { Lowercase = true };
            _tinyNNFactory = new TinyNNModelFactory();
            _modelKind = "tinynn";
        }

        [TestCase("../../../../../data/showcase.txt")]
        [TestCase("../../../../../data/showcase2.txt")]
        [TestCase("../../../../../data/showcase3.txt")]
        public void TrainingPipeline_FromTextFile_ToCheckpoint_CompletesSuccessfully(string pathToLoad)
        {
            Corpus corpus = _loader.Load(pathToLoad, _loadOptions);
            WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            int[] tokens = tokenizer.Encode(corpus.TrainText);

            TinyNNConfig tinyNNConfig = new TinyNNConfig(tokenizer.VocabSize);
            TinyNNWeights tinyNNWeights = new TinyNNWeights(tinyNNConfig.VocabSize, tinyNNConfig.EmbeddingSize);
            ILanguageModel model = new TinyNNModel(_modelKind, tinyNNConfig.VocabSize, tinyNNConfig, tinyNNWeights);

            ITokenStream tokenStream = new ArrayTokenStream(tokens);
            BatchWindowSampler windowSampler = new BatchWindowSampler();
            IBatchProvider batchProvider = new TokenBatchProvider(tokenStream, windowSampler);

            _training.Train(model, batchProvider, _trainingConfig, _batchConfig, null, "../../../../../data/checkpoints/TinyNNCheckpoints.json", "word", tokenizer, seed);

            float[] scoresBefore = model.NextTokenScores(tokens);

            string path = "TinyNNCheckpoints.json";

            Checkpoint checkpoint = new Checkpoint(
                ModelKind: model.ModelKind,
                TokenizerKind: "word",
                TokenizerPayload: JsonSerializer.SerializeToElement(tokenizer.GetPayloadForCheckpoint()),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: 42,
                ContractFingerprintChain: $"{tokenizer.GetContractFingerprint()} and {model.GetContractFingerprint()}"
            );

            JsonCheckpointIO.Save(path, checkpoint);
            Checkpoint loaded = JsonCheckpointIO.Load(path);

            JsonElement payload = (JsonElement)loaded.ModelPayload;

            TinyNNModel modelAfter = (TinyNNModel)_tinyNNFactory.CreateFromPayload(payload, _modelKind);
            float[] scoresAfter = modelAfter.NextTokenScores(tokens);

            Assert.That(modelAfter.ModelKind, Is.EqualTo("tinynn"));
            Assert.That(modelAfter.Config, Is.Not.Null);
            Assert.That(modelAfter.Weights, Is.Not.Null);
            Assert.That(scoresBefore, Is.EqualTo(scoresAfter));
        }
    }
}