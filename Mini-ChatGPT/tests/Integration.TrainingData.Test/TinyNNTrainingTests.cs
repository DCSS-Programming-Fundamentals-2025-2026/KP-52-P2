using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Streams;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Training;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Contracts;

namespace Integration.TrainingData.Test
{
    public class TinyNNTrainingTests
    {
        private TinyNNConfig _tinyNNConfig;
        private TinyNNWeights _tinyNNWeights;
        private ArrayTokenStream _arrayTokenStream;
        private TrainingLoop _trainingLoop;

        private int _vocabSize;
        private string _modelKind;

        private int[] _tokens;
                                                        
        [SetUp]
        public void Setup()
        {
            _vocabSize = 4;
            _modelKind = "tinynn";
            _tokens = [0, 1, 2, 3];

            _trainingLoop = new TrainingLoop();
            _arrayTokenStream = new ArrayTokenStream(_tokens);
            _tinyNNConfig = new TinyNNConfig(_vocabSize);
            _tinyNNWeights = new TinyNNWeights(_tinyNNConfig.VocabSize, _tinyNNConfig.EmbeddingSize);
        }

        [Test]
        public void BatchingAndTraining_TinyNN_CanRunTraining()
        {
            ILanguageModel model = new TinyNNModel(_modelKind, _vocabSize, _tinyNNConfig, _tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            TrainingMetrics metrics = _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null, "../../../../../data/checkpoints/TinyNNCheckpoints.json");

            Assert.That(metrics.AverageLoss, Is.Not.NaN);
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(10));
            Assert.That(metrics.ElapsedTime, Is.Not.Default);
            Assert.That(metrics.TotalSteps, Is.EqualTo(20));
        }

        [Test]
        public void BatchingAndTraining_NullModel_ThrowsArgumentException()
        {
            ILanguageModel? model = null;
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            Assert.Throws<ArgumentException>(() => _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null, "../../../../../data/checkpoints/TinyNNCheckpoints.json")); 
        }
    }
}