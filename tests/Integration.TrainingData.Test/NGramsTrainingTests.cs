using Contracts;
using Lib.Models.NGram;
using Lib.Training;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Integration.TrainingData.Test;

public class NGramsTrainingTests
{
    private int? seed;

    [SetUp]
    public void Setup()
    {
        seed = 42;   
    }

    [Test]
    public void BatchingAndTraining_CanRunOneEpoch_Bigram()
    {
        // Arrange
        int[] tokens = new int[] { 1, 2, 3, 2, 1, 2, 3};
        ILanguageModel model = new NGramModel(tokens.Length);

        Checkpoint checkpoint = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");
        string effectiveTokenizerKind = checkpoint.TokenizerKind.ToLowerInvariant();
        ITokenizer tokenizer = MiniChatGPT.App.Program.RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);

        var trainingConfig = new TrainingConfig(1, 0.01f, 1);
        TrainingLoop trainingLoop = new TrainingLoop();

        // Act
        TrainingMetrics metrics = trainingLoop.Train(model, null, trainingConfig, null, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", effectiveTokenizerKind, tokenizer, seed);

        // Assert
        Assert.That(metrics, Is.Not.Null);
        Assert.That(metrics.Perplexity, Is.GreaterThan(0));
        Assert.That(float.IsFinite((float)metrics.Perplexity));
        Assert.That(metrics.NGramCount, Is.EqualTo(6));
        Assert.That(metrics.CurrentEpoch, Is.EqualTo(1));
    }

    [Test]
    public void BatchingAndTraining_CanRunOneEpoch_Trigram()
    {
        // Arrange
        int[] tokens = new int[] { 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1 };
        ILanguageModel model = new NGramModel(tokens.Length);

        var trainingConfig = new TrainingConfig(1, 0.01f, 1);
        TrainingLoop trainingLoop = new TrainingLoop();

        Checkpoint checkpoint = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");
        string effectiveTokenizerKind = checkpoint.TokenizerKind.ToLowerInvariant();
        ITokenizer tokenizer = MiniChatGPT.App.Program.RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);

        // Act
        TrainingMetrics metrics = trainingLoop.Train(model, null, trainingConfig, null, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", effectiveTokenizerKind, tokenizer, seed);

        // Assert
        Assert.That(metrics, Is.Not.Null);
        Assert.That(metrics.Perplexity, Is.GreaterThan(0));
        Assert.That(float.IsFinite((float)metrics.Perplexity));
        Assert.That(metrics.NGramCount, Is.EqualTo(12));
        Assert.That(metrics.CurrentEpoch, Is.EqualTo(1));
    }

    [Test]
    public void BatchingAndTraining_Trigram_EpochsCompare()
    {
        // Arrange
        int[] tokens = new int[] { 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1 };
        ILanguageModel model = new NGramModel(tokens.Length);

        var trainingConfig1 = new TrainingConfig(1, 0.01f, 1);
        var trainingConfig2 = new TrainingConfig(20, 0.01f, 5);
        TrainingLoop trainingLoop = new TrainingLoop();

        Checkpoint checkpoint = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");
        string effectiveTokenizerKind = checkpoint.TokenizerKind.ToLowerInvariant();
        ITokenizer tokenizer = MiniChatGPT.App.Program.RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);

        // Act
        TrainingMetrics metrics1 = trainingLoop.Train(model, null, trainingConfig1, null, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", effectiveTokenizerKind, tokenizer, seed);
        TrainingMetrics metrics2 = trainingLoop.Train(model, null, trainingConfig2, null, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", effectiveTokenizerKind, tokenizer, seed);

        // Assert
        Assert.That(metrics1, Is.Not.Null);
        Assert.That(metrics2, Is.Not.Null);
        Assert.That(metrics1.Perplexity, Is.EqualTo(metrics2.Perplexity));
        Assert.That(metrics1.NGramCount, Is.EqualTo(metrics2.NGramCount));
        Assert.That(metrics1.CurrentEpoch, Is.EqualTo(1));
        Assert.That(metrics2.CurrentEpoch, Is.EqualTo(20));
    }

    [Test]
    public void BatchingAndTraining_NGram_InvalidToken()
    {
        // Arrange
        int[] tokens = new int[] { 1, 2, 3, 2, -3, 1, 2, 3, 1, 2, 3, 2, 1 };
        ILanguageModel model = new NGramModel(tokens.Length);

        var trainingConfig = new TrainingConfig(20, 0.01f, 5);
        TrainingLoop trainingLoop = new TrainingLoop();

        Checkpoint checkpoint = JsonCheckpointIO.Load("../../../../../data/checkpoints/NGramCheckpoints.json");
        string effectiveTokenizerKind = checkpoint.TokenizerKind.ToLowerInvariant();
        ITokenizer tokenizer = MiniChatGPT.App.Program.RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);

        // Act + Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => trainingLoop.Train(model, null, trainingConfig, null, tokens, "../../../../../data/checkpoints/NGramCheckpoints.json", effectiveTokenizerKind, tokenizer, seed));
    }
}
