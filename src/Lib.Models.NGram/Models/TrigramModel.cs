using Contracts;
using System.Text.Json;
using Lib.Models.NGram.Serialization;

public class TrigramModel : ILanguageModel, INGramModel
{
    private const int UnknownTokenId = 0;
    private const float ComparisonEpsilon = 0.000001f;

    public Dictionary<(int, int), float[]> _trigramProbs = new Dictionary<(int, int), float[]>();
    public NGramModel bigramModel;
    private NGramCounts counts = new NGramCounts();
    public int VocabSize { get; set; }
    public string ModelKind { get; }

    public TrigramModel(int vocabSize)
    {
        VocabSize = vocabSize;
        counts = new NGramCounts();
        bigramModel = new NGramModel(VocabSize);
        ModelKind = "trigram";

        _trigramProbs = CreateEmptyTrigramDictionary(VocabSize);
    }

    public string GetContractFingerprint()
    {
        return $"Lib.Models.NGram:1.0.0:NGramModel,TrigramModel,NGramModelFactory";
    }

    public override bool Equals(object? obj)
    {
        if (obj == null)
        {
            return false;
        }

        if (obj is not TrigramModel model)
        {
            return false;
        }

        if (model.ModelKind != ModelKind ||
            model.VocabSize != VocabSize ||
            !model.bigramModel.Equals(bigramModel))
        {
            return false;
        }

        foreach (var pair in _trigramProbs)
        {
            if (!model._trigramProbs.TryGetValue(pair.Key, out float[]? otherRow))
            {
                return false;
            }

            for (int i = 0; i < VocabSize; i++)
            {
                if (Math.Abs(otherRow[i] - pair.Value[i]) > ComparisonEpsilon)
                {
                    return false;
                }
            }
        }

        return true;
    }
    public void ResetTrainingResults()
    {
        counts = new NGramCounts();

        if (bigramModel != null)
        {
            bigramModel.ResetTrainingResults();
        }
        else
        {
            bigramModel = new NGramModel(VocabSize);
        }

        _trigramProbs = CreateEmptyTrigramDictionary(VocabSize);
    }

    public void Train(ReadOnlySpan<int> tokens)
    {
        ResetTrainingResults();
        bigramModel.Train(tokens);


        if (tokens.Length < 3)
        {
            return;
        }

        counts.CountTrigrams(_trigramProbs, tokens);

        foreach (var bigram in _trigramProbs)
        {
            int countForPair = counts.GetTrigramPrevsTotal(_trigramProbs, bigram.Key);

            if (countForPair != 0)
            {
                for (int j = 0; j < bigram.Value.Length; j++)
                {
                    bigram.Value[j] = bigram.Value[j] / countForPair;
                }
            }
        }
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        if (context.Length < 2)
        {
            return bigramModel.NextTokenScores(context);
        }

        int lastToken = context[context.Length - 1];
        int beforeLastToken = context[context.Length - 2];

        if (lastToken < 0 || lastToken >= VocabSize
            || beforeLastToken < 0 || beforeLastToken >= VocabSize)
        {
            return bigramModel.NextTokenScores(context);
        }

        bool hasAnyObservedTransition = false;

        for (int i = 0; i < VocabSize; i++)
        {
            if (_trigramProbs[(beforeLastToken, lastToken)][i] != 0)
            {
                hasAnyObservedTransition = true;
                break;
            }
        }

        if (hasAnyObservedTransition)
        {
            float[] result = (float[])_trigramProbs[(beforeLastToken, lastToken)].Clone();
            result[0] = 0f;
            return result;
        }

        return bigramModel.NextTokenScores(context);
    }

    public void FromPayload(JsonElement json)
    {
        NGramPayloadMapper mapper = new NGramPayloadMapper();
        mapper.FromJsonElementToTrigram(json, this);
    }

    public object GetPayloadForCheckpoint()
    {
        NGramPayloadMapper mapper = new NGramPayloadMapper();
        return mapper.FromTrigramToJson(this);
    }

    private static Dictionary<(int, int), float[]> CreateEmptyTrigramDictionary(int vocabSize)
    {
        Dictionary<(int, int), float[]> result = new Dictionary<(int, int), float[]>();

        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < vocabSize; j++)
            {
                result[(i, j)] = new float[vocabSize];
            }
        }

        return result;
    }
}