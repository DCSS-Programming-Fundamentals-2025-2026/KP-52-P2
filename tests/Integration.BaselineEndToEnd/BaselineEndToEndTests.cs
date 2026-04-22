using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Tokenization.Application;
using Lib.Tokenization.Domain.Model;
using Lib.Tokenization.Infrastructure.Serialization;
using Contracts;
using System.Text.Json;
using Lib.Models.Trigram;
using Lib.Models.NGram.Factories;
using Lib.Models.NGram;

public class BaselineEndToEndTests
{
    [Test]
    public void TrigramOnRealCorpus_GeneratesUkrainianWords()
    {
        // Arrange
        CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
        Corpus corpus = loader.Load("../../../../../data/showcase.txt", new CorpusLoadOptions { Lowercase = true });

        WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
        int[] tokens = tokenizer.Encode(corpus.TrainText);

        TrigramModel model = new TrigramModel(tokenizer.VocabSize);
        model.Train(tokens);

        // Act
        List<int> context = new List<int>
        {
            tokens[2],
            tokens[3]
        };

        for (int i = 0; i < 10; i++)
        {
            float[] scores = model.NextTokenScores(context.ToArray());
            context.Add(Array.IndexOf(scores, scores.Max()) + 1);
        }

        string text = tokenizer.Decode(context.ToArray());
        Console.WriteLine(text);

        // Assert
        Assert.That(text, Is.Not.Null);
        Assert.That(text.Length, Is.GreaterThan(5));
    }

    [Test]
    public void FullCheckpoint_SaveLoad_SameGeneration()
    {
        // Arrange
        string text =
        "Калина похилилася журиться калина похилилася журиться " +
        "сонце світить калина похилилася журиться";

        var corpusLoader = new CorpusLoader(new DefaultFileSystem());
        var corpus = corpusLoader.LoadFromText(text, new CorpusLoadOptions(Lowercase: true));

        var tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
        int[] tokens = tokenizer.Encode(corpus.TrainText);

        var model = new TrigramModel(tokenizer.VocabSize);
        model.Train(tokens);

        string path = Path.Combine(Path.GetTempPath(), "checkpoint_test.json");

        var checkpoint = new Checkpoint(
            ModelKind: model.ModelKind,
            TokenizerKind: "word",
            TokenizerPayload: JsonSerializer.SerializeToElement(tokenizer.GetPayloadForCheckpoint()),
            ModelPayload: model.GetPayloadForCheckpoint(),
            Seed: 42,
            ContractFingerprintChain: $"{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
        );

        // Act
        JsonCheckpointIO.Save(path, checkpoint);

        var loaded = JsonCheckpointIO.Load(path);

        JsonElement tokenizerPayload = (JsonElement)loaded.TokenizerPayload;

        if (tokenizerPayload.ValueKind == JsonValueKind.Object)
        {
            if (tokenizerPayload.TryGetProperty("Words", out JsonElement wordsElement))
            {
                string[] words = wordsElement.Deserialize<string[]>() ?? Array.Empty<string>();
                tokenizerPayload = JsonSerializer.SerializeToElement(new { Words = words });
            }
            else if (tokenizerPayload.TryGetProperty("Words", out JsonElement wordsLowerElement))
            {
                string[] words = wordsLowerElement.Deserialize<string[]>() ?? Array.Empty<string>();
                tokenizerPayload = JsonSerializer.SerializeToElement(new { Words = words });
            }
            else
            {
                throw new InvalidOperationException("Tokenizer payload does not contain Words/words.");
            }
        }

        var restoredTokenizer = TokenizerPayloadSerializer.RestoreTokenizer(
            loaded.TokenizerKind,
            tokenizerPayload
        );

        var factory = new NGramModelFactory();
        var restoredModel = factory.Create(loaded.ModelKind, restoredTokenizer.VocabSize);

        if (restoredModel is TrigramModel trigram)
        {
            trigram.FromPayload((JsonElement)loaded.ModelPayload);
        }
        else if (restoredModel is NGramModel bigram)
        {
            bigram.FromPayload((JsonElement)loaded.ModelPayload);
        }

        // генерація ДО
        List<int> context1 = new List<int> { tokens[0], tokens[1] };

        for (int i = 0; i < 10; i++)
        {
            float[] scores = model.NextTokenScores(context1.ToArray());
            int next = Array.IndexOf(scores, scores.Max());
            context1.Add(next);
        }

        string originalText = tokenizer.Decode(context1.ToArray());

        // генерація ПІСЛЯ
        List<int> context2 = new List<int> { tokens[0], tokens[1] };

        for (int i = 0; i < 10; i++)
        {
            float[] scores = restoredModel.NextTokenScores(context2.ToArray());
            int next = Array.IndexOf(scores, scores.Max());
            context2.Add(next);
        }

        string restoredText = restoredTokenizer.Decode(context2.ToArray());

        Console.WriteLine("Original:");
        Console.WriteLine(originalText);

        Console.WriteLine("Restored:");
        Console.WriteLine(restoredText);

        // Assert
        Assert.That(originalText, Is.Not.Empty);
        Assert.That(restoredText, Is.Not.Empty);
        Assert.That(restoredText, Is.EqualTo(originalText));
    }

    [Test]
    public void WordAndCharTokenizer_BothProduceReadableOutput()
    {
        // Arrange
        CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
        Corpus corpus = loader.Load("../../../../../data/showcase3.txt", new CorpusLoadOptions { Lowercase = true });

        WordTokenizer wordTokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
        CharTokenizer charTokenizer = CharTokenizer.BuildFromText(corpus.TrainText);

        int[] wordTokens = wordTokenizer.Encode(corpus.TrainText);
        int[] charTokens = charTokenizer.Encode(corpus.TrainText);

        TrigramModel wordModel = new TrigramModel(wordTokenizer.VocabSize);
        wordModel.Train(wordTokens);

        TrigramModel charModel = new TrigramModel(charTokenizer.VocabSize);
        charModel.Train(charTokens);

        // Act
        List<int> wordContext = new List<int>
        {
            wordTokens[0],
            wordTokens[1]
        };

        List<int> charContext = new List<int>
        {
            charTokens[0],
            charTokens[1]
        };

        for (int i = 0; i < 10; i++)
        {
            float[] wordScores = wordModel.NextTokenScores(wordContext.ToArray());
            wordContext.Add(Array.IndexOf(wordScores, wordScores.Max()) + 1);
        }

        for (int i = 0; i < 30; i++)
        {
            float[] charScores = charModel.NextTokenScores(charContext.ToArray());
            charContext.Add(Array.IndexOf(charScores, charScores.Max()) + 1);
        }

        string wordText = wordTokenizer.Decode(wordContext.ToArray());
        Console.WriteLine("Words: " + wordText);

        string charText = charTokenizer.Decode(charContext.ToArray());
        Console.WriteLine("Chars: " + charText);

        // Assert
        Assert.That(wordText, Is.Not.Null);
        Assert.That(charText, Is.Not.Null);
        Assert.That(wordText.Length, Is.GreaterThan(5));
        Assert.That(charText.Length, Is.GreaterThan(20));
    }

    [Test]
    public void Trigram_Perplexity_OnValidation_IsFinite()
    {
        // Arrange
        CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
        Corpus corpus = loader.Load("../../../../../data/showcase2.txt", new CorpusLoadOptions { Lowercase = true });

        WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
        int[] trainTokens = tokenizer.Encode(corpus.TrainText);
        int[] validationTokens = tokenizer.Encode(corpus.ValText);

        TrigramModel model = new TrigramModel(tokenizer.VocabSize);
        model.Train(trainTokens);

        PerplexityCalculator calculator = new PerplexityCalculator();

        // Act
        float perplexity = calculator.ComputePerplexityTrigram(model, validationTokens);

        // Assert
        Assert.That(perplexity, Is.GreaterThan(0));
        Assert.That(float.IsFinite(perplexity), Is.True);
    }

    [Test]
    public void Trigram_HasLowerPerplexity_ThanBigram()
    {
        // Arrange
        CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
        Corpus corpus = loader.Load("../../../../../data/showcase3.txt", new CorpusLoadOptions { Lowercase = true });

        WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
        int[] trainTokens = tokenizer.Encode(corpus.TrainText);
        int[] validationTokens = tokenizer.Encode(corpus.ValText);

        PerplexityCalculator calculator = new PerplexityCalculator();

        TrigramModel trigram = new TrigramModel(tokenizer.VocabSize);
        trigram.Train(trainTokens);

        NGramModel bigram = new NGramModel(tokenizer.VocabSize);
        bigram.Train(trainTokens);

        // Act
        float perplexityTrigram = calculator.ComputePerplexityTrigram(trigram, validationTokens);
        float perplexityBigram = calculator.ComputePerplexityBigram(bigram, validationTokens);

        // Assert
        Assert.That(perplexityTrigram, Is.GreaterThan(0));
        Assert.That(perplexityBigram, Is.GreaterThan(0));
        Assert.That(perplexityBigram, Is.GreaterThan(perplexityTrigram));
    }

    [Test]
    public void NextTokenScores_SumsToOne_ForAllTokens()
    {
        // Arrange
        CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
        Corpus corpus = loader.Load("../../../../../data/short_input.txt", new CorpusLoadOptions { Lowercase = true });

        WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);

        int[] tokens = tokenizer.Encode(corpus.TrainText);

        TrigramModel model = new TrigramModel(tokenizer.VocabSize);
        model.Train(tokens);

        // Act
        float[] probs = model.NextTokenScores(new int[] { tokens[0], tokens[1] });
        float sum = probs[0] + probs[1] + probs[2] + probs[3] + probs[4] + probs[5] + probs[6] + probs[7] + probs[8];

        // Assert
        Assert.That(sum, Is.EqualTo(1).Within(1e-5));
    }
}