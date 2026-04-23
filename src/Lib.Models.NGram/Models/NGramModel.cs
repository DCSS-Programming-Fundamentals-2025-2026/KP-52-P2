using Contracts;
using System.Text.Json;
using Lib.Models.NGram.Serialization;

public class NGramModel : ILanguageModel, INGramModel
{
    private const int UnknownTokenId = 0;
    private const float ComparisonEpsilon = 0.000001f;

    public float[][] _probs; 
    public int VocabSize { get; set; }
    private NGramCounts counts = new NGramCounts();
    public string ModelKind { get; }

    public NGramModel(int vocabSize) 
    {
        VocabSize = vocabSize;
        _probs = new float[vocabSize][];
        ModelKind = "bigram";

        for (int i = 0; i < vocabSize; i++)
        {
            _probs[i] = new float[vocabSize];
        }
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

        if (obj is not NGramModel model)
        {
            return false;
        }

        if (model.ModelKind != ModelKind || model.VocabSize != VocabSize)
        {
            return false;
        }

        for (int i = 0; i < VocabSize; i++)
        {
            for (int j = 0; j < VocabSize; j++)
            {
                if(Math.Abs(model._probs[i][j] - _probs[i][j]) > ComparisonEpsilon)
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

        _probs = new float[VocabSize][];
        for (int i = 0; i < VocabSize; i++)
        {
            _probs[i] = new float[VocabSize];
        }
    }

    public void Train(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length < 2)
        {
            return;
        }

        ResetTrainingResults();
        counts.CountBigrams(_probs, tokens);

        for (int i = 0; i < VocabSize; i++)
        {
            int countForRow = counts.GetBigramPrevTotal(_probs, VocabSize, i);

            if (countForRow != 0)
            {
                for (int j = 0; j < VocabSize; j++)
                {
                    _probs[i][j] = _probs[i][j] / countForRow;
                }
            }
        }
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        float[] alternative = new float[VocabSize];

        if (VocabSize > 1)
        {
            float uniform = 1f / (VocabSize - 1);

            for (int i = 1; i < VocabSize; i++)
            {
                alternative[i] = uniform;
            }
        }

        alternative[UnknownTokenId] = 0f;

        if (context.Length < 1)
        {
            return alternative;
        }

        int lastToken = context[context.Length - 1];

        if (lastToken < 0 || lastToken >= VocabSize)
        {
            return alternative;
        }

        bool hasAnyObservedTransition = false;

        for (int i = 0; i < VocabSize; i++)
        {
            if (_probs[lastToken][i] != 0)
            {
                hasAnyObservedTransition = true;
                break;
            }
        }

        if (hasAnyObservedTransition)
        {
            float[] result = (float[])_probs[lastToken].Clone();
            result[0] = 0f;
            return result;
        }

        return alternative;
    }

    public void FromPayload(JsonElement json)
    {
        NGramPayloadMapper mapper = new NGramPayloadMapper();
        mapper.FromJsonElementToBigram(json, this);
    }

    public object GetPayloadForCheckpoint()
    {
        NGramPayloadMapper mapper = new NGramPayloadMapper();      
        return mapper.FromBigramToJson(this);
    }
}