using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Contracts;

namespace Lib.Training;

public class TrainingLoop : ITrainingLoop
{
    public TrainingMetrics Train (ILanguageModel model, IBatchProvider batchProvider,TrainingConfig config, BatchConfig batchConfig,
        int[] tokens, string checkpointPath, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed)
    {
        if (model == null)
        {
            throw new ArgumentException("Invalid data");
        }

        TrainingLoopImpl loopImpl = new TrainingLoopImpl();
        if (model.ModelKind == "bigram" || model.ModelKind == "trigram")
        {                                                                                                 
            return loopImpl.TrainNGram(model, tokens, config, checkpointPath, effectiveTokenizerKind, tokenizer, seed);
        }
        else if (model.ModelKind == "tinynn")
        {
            return loopImpl.TrainTinyNN(model, batchProvider, config, batchConfig, checkpointPath, effectiveTokenizerKind, tokenizer, seed);
        }                        
        else if (model.ModelKind == "tinytransformer")
        {
            return loopImpl.TrainTransformer(model, batchProvider, config, batchConfig, checkpointPath, effectiveTokenizerKind, tokenizer, seed);
        }

        throw new ArgumentException("Invalid data");
    }
}