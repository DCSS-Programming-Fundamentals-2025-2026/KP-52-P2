using Contracts;
using Lib.Models.NGram;
using Lib.Models.Trigram;
using System.Text.Json;
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
        var jsonElement = (JsonElement)payload;
        var modelPayload = jsonElement.GetProperty("modelPayload");
        
        var bigramProbs = modelPayload.GetProperty("bigramProbs");
        int vocabSize = bigramProbs.GetArrayLength(); 

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