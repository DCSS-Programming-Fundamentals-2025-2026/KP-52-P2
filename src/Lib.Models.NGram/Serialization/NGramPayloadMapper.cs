using System.Text.Json;

namespace Lib.Models.NGram.Serialization;
public class NGramPayloadMapper
{
    public JsonElement FromBigramToJson(NGramModel model)
    {
        object obj = new
        {
            bigramProbs = model._probs
        };

        return JsonSerializer.SerializeToElement(obj); ;
    }

    public void FromJsonElementToBigram(JsonElement jsonElement, NGramModel model)
    {
        if (!jsonElement.TryGetProperty("bigramProbs", out JsonElement probsElement))
        {
            throw new InvalidOperationException("Checkpoint is missing bigramProbs.");
        }

        float[][]? probs = JsonSerializer.Deserialize<float[][]>(probsElement.GetRawText());

        if (probs == null || probs.Length == 0)
        {
            throw new InvalidOperationException("Failed to restore bigram probabilities.");
        }

        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] == null || probs[i].Length != probs.Length)
            {
                throw new InvalidOperationException("Invalid bigram probability matrix shape.");
            }
        }

        model._probs = probs;
        model.VocabSize = probs.Length;
    }

    public JsonElement FromTrigramToJson(TrigramModel model)
    {
        Dictionary<string, float[]> temp = new Dictionary<string, float[]>();
        foreach (var pair in model._trigramProbs)
        {
            string newKey = $"{pair.Key.Item1}, {pair.Key.Item2}";
            temp.Add(newKey, pair.Value);
        }

        object obj = new
        {
            bigramProbs = model.bigramModel._probs,
            trigramProbs = temp
        };

        return JsonSerializer.SerializeToElement(obj);
    }

    public void FromJsonElementToTrigram(JsonElement jsonElement, TrigramModel model)
    {
        if (!jsonElement.TryGetProperty("bigramProbs", out JsonElement bigramElement))
        {
            throw new InvalidOperationException("Checkpoint is missing bigramProbs.");
        }

        if (!jsonElement.TryGetProperty("trigramProbs", out JsonElement trigramElement))
        {
            throw new InvalidOperationException("Checkpoint is missing trigramProbs.");
        }

        float[][]? bigramProbs = JsonSerializer.Deserialize<float[][]>(bigramElement.GetRawText());
        if (bigramProbs == null || bigramProbs.Length == 0)
        {
            throw new InvalidOperationException("Failed to restore bigram probabilities.");
        }

        for (int i = 0; i < bigramProbs.Length; i++)
        {
            if (bigramProbs[i] == null || bigramProbs[i].Length != bigramProbs.Length)
            {
                throw new InvalidOperationException("Invalid bigram probability matrix shape.");
            }
        }

        Dictionary<string, float[]>? temp = JsonSerializer.Deserialize<Dictionary<string, float[]>>(trigramElement.GetRawText());
        if (temp == null)
        {
            throw new InvalidOperationException("Failed to restore trigram probabilities.");
        }

        int vocabSize = bigramProbs.Length;

        model.bigramModel = new NGramModel(vocabSize);
        model.bigramModel._probs = bigramProbs;
        model.bigramModel.VocabSize = vocabSize;

        model.VocabSize = vocabSize;
        model._trigramProbs = CreateEmptyTrigramDictionary(vocabSize);

        foreach (var pair in temp)
        {
            string[] keys = pair.Key.Split(',');
            if (keys.Length != 2)
            {
                continue;
            }

            if (!int.TryParse(keys[0].Trim(), out int first))
            {
                continue;
            }

            if (!int.TryParse(keys[1].Trim(), out int second))
            {
                continue;
            }

            if (first < 0 || first >= vocabSize || second < 0 || second >= vocabSize)
            {
                continue;
            }

            if (pair.Value == null || pair.Value.Length != vocabSize)
            {
                continue;
            }

            model._trigramProbs[(first, second)] = (float[])pair.Value.Clone();
        }
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