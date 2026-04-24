using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Contracts;

namespace Lib.Training;

public interface ITrainingLoop
{
    TrainingMetrics Train(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config, BatchConfig batchConfig,
        int[] tokens, string checkpointPath, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed);
}