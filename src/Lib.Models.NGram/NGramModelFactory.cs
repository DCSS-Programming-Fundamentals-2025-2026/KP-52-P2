using Contracts;
using Lib.Models.NGram.Serialization;
using System.Text.Json;

namespace Lib.Models.NGram.Factories;

public class NGramModelFactory : INGramModelFactory
{
    public ILanguageModel Create(string modelType, int vocabSize)
    {
        if (modelType == "bigram")
        {
            return new NGramModel(vocabSize);
        }

        if (modelType == "trigram")
        {
            return new TrigramModel(vocabSize);
        }

        throw new ArgumentException($"Unknown model type: {modelType}");
    }

    public ILanguageModel CreateFromPayload(string modelType, object payload)
    {
        if (payload is not JsonElement jsonElement)
            throw new ArgumentException("Payload must be a JsonElement");


        if (!jsonElement.TryGetProperty("bigramProbs", out JsonElement bigramElement))
            throw new InvalidOperationException("Invalid JSON: missing bigramProbs");


        int vocabSize = bigramElement.GetArrayLength();

        if (modelType == "bigram")
        {
            var model = new NGramModel(vocabSize);
            var mapper = new NGramPayloadMapper();
            mapper.FromJsonElementToBigram(jsonElement, model);
            return model;
        }

        if (modelType == "trigram")
        {
            var model = new TrigramModel(vocabSize);
            var mapper = new NGramPayloadMapper();
            mapper.FromJsonElementToTrigram(jsonElement, model);
            return model;
        }

        throw new ArgumentException($"Unknown model type: {modelType}");
    }
}