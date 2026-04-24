using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Models.NGram;
using Lib.Models.TinyNN;
using Lib.Models.TinyTransformer;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Lib.Training.Scheduling;
using System.Reflection;
using System.Text.Json;

namespace Lib.Training;

public class TrainingLoopImpl 
{
    public TrainingMetrics TrainTinyNN(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config,
        BatchConfig batchConfig, string checkpointPath, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed)
    {
        if (model is not TinyNNModel tinyNNModel)
            throw new InvalidCastException("Invalid model");

        TrainingMetrics metrics = new TrainingMetrics();
        Random rng = new Random();

        int totalSteps = 0;
        for (int i = 0; i < config.Epochs; i++)
        {
            float sumLoss = 0f;
            int counter = 0;
            DateTime startTime = DateTime.Now;

            Batch batch = batchProvider.GetBatch(batchConfig, rng);
            int[][] inputs = batch.Inputs;
            int[] targets = batch.Targets;

            for (int j = 0; j < inputs.Length; j++)
            {
                int[] context = inputs[j];
                int target = targets[j];

                float loss = tinyNNModel.TrainStep(context, target, config.LearningRate);
                sumLoss += loss;
                counter++;
            }

            totalSteps += counter;

            if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
            {
                DateTime finishTime = DateTime.Now;
                TimeSpan delta = finishTime - startTime;

                float averageLoss = 0f;
                if (counter != 0)
                {
                    averageLoss = sumLoss / counter;
                }

                metrics.UpdateNN(i + 1, averageLoss, totalSteps, delta);

                SaveCheckpoint(model, effectiveTokenizerKind, tokenizer, seed, checkpointPath);
            }
        }

        return metrics;
    }

    public TrainingMetrics TrainNGram(ILanguageModel model, int[] tokens, TrainingConfig config,
        string checkpointPath, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed)
    {
        int n;
        INGramModel nGramModel;

        if (model.ModelKind == "bigram" && model is NGramModel bigramModel)
        {
            nGramModel = bigramModel;
            n = 2;
        }
        else if (model.ModelKind == "trigram" && model is TrigramModel trigramModel)
        {
            nGramModel = trigramModel;
            n = 3;
        }
        else
        {
            throw new InvalidCastException("Invalid model");
        }

        TrainingMetrics metrics = new TrainingMetrics();

        for (int i = 0; i < config.Epochs; i++)
        {
            if (i == config.Epochs - 1)
            {
                float perplexity = 0f;
                int nGramCount = 0;

                DateTime startTime = DateTime.Now;

                nGramModel.Train(tokens);

                DateTime finishTime = DateTime.Now;
                TimeSpan delta = finishTime - startTime;

                PerplexityCalculator calculator = new PerplexityCalculator();

                if (n == 2)
                {
                    perplexity = calculator.ComputePerplexityBigram((NGramModel)nGramModel, tokens);
                }
                else
                {
                    perplexity = calculator.ComputePerplexityTrigram((TrigramModel)nGramModel, tokens);
                }

                nGramCount = tokens.Length - n + 1;
                if (nGramCount < 0)
                {
                    nGramCount = 0;
                }

                metrics.UpdateNGram(i + 1, perplexity, nGramCount, delta);

                if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
                {
                    SaveCheckpoint(model, effectiveTokenizerKind, tokenizer, seed, checkpointPath);
                }
            }
        }

        return metrics;
    }

    public TrainingMetrics TrainTransformer(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config,
        BatchConfig batchConfig, string checkpointPath, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed)
    {
        if (model is not TinyTransformerModel tinyTransformerModel)
            throw new InvalidCastException("Invalid model");

        TrainingMetrics metrics = new TrainingMetrics();
        Random rng = new Random();

        int totalSteps = 0;
        for (int i = 0; i < config.Epochs; i++)
        {
            float sumLoss = 0f;
            int counter = 0;
            DateTime startTime = DateTime.Now;

            Batch batch = batchProvider.GetBatch(batchConfig, rng);
            int[][] inputs = batch.Inputs;
            int[] targets = batch.Targets;

            for (int j = 0; j < inputs.Length; j++)
            {
                int[] context = inputs[j];
                int target = targets[j];

                float loss = tinyTransformerModel.TrainStep(context, target, config.LearningRate);
                sumLoss += loss;
                counter++;
            }

            totalSteps += counter;

            if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
            {
                DateTime finishTime = DateTime.Now;
                TimeSpan delta = finishTime - startTime;

                float averageLoss = 0f;
                if (counter != 0)
                {
                    averageLoss = sumLoss / counter;
                }

                metrics.UpdateNN(i + 1, averageLoss, totalSteps, delta);  

                SaveCheckpoint(model, effectiveTokenizerKind, tokenizer, seed, checkpointPath);
            }
        }

        return metrics;
    }

    private void SaveCheckpoint (ILanguageModel model, string effectiveTokenizerKind, ITokenizer tokenizer, int? seed, string checkpointPath)
    {
        var jsonElement = model.GetPayloadForCheckpoint();
        string json = JsonSerializer.Serialize(jsonElement, new JsonSerializerOptions { WriteIndented = true });

        File.WriteAllText(checkpointPath, json);

        var checkpointToSave = new Checkpoint(
                    ModelKind: model.ModelKind,
                    TokenizerKind: effectiveTokenizerKind,
                    TokenizerPayload: JsonSerializer.SerializeToElement(tokenizer.GetPayloadForCheckpoint()),
                    ModelPayload: model.GetPayloadForCheckpoint(),
                    Seed: seed ?? 42,
                    ContractFingerprintChain: model.GetContractFingerprint()
                );

        JsonCheckpointIO.Save(checkpointPath, checkpointToSave);
    }
}