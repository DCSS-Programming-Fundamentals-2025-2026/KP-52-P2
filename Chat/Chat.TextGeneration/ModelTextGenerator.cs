using Contracts;
using Lib.MathCore;
using Lib.Sampling;

namespace Chat.TextGeneration;

public class ModelTextGenerator : ITextGenerator
{
    private readonly ILanguageModel model;
    private readonly ITokenizer tokenizer;
    private readonly ISampler sampler;

    public ModelTextGenerator(
        ILanguageModel model,
        ITokenizer tokenizer,
        ISampler sampler)
    {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
    }

    public string Generate(
        string prompt,
        int maxTokens,
        float temperature,
        int topK,
        int? seed = null)
    {
        List<int> context = tokenizer.Encode(prompt).ToList();
        List<int> generated = new List<int>();

        if (context.Count == 0)
        {
            context.Add(0);
        }

        Random? rng = seed.HasValue ? new Random(seed.Value) : null;

        for (int i = 0; i < maxTokens; i++)
        {
            float[] scores = model.NextTokenScores(context.ToArray());

            float[] probs;

            if (model.ModelKind == "tinynn" ||
                model.ModelKind == "tinytransformer")
            {
                probs = MathOps.Default.Softmax(scores);
            }
            else
            {
                probs = scores;
            }

            ApplyRepetitionPenalty(probs, generated);

            int nextToken = sampler.Sample(
                probs,
                temperature,
                topK,
                rng);

            context.Add(nextToken);
            generated.Add(nextToken);
            
            string decodedText = tokenizer.Decode(new[] { nextToken });
            Console.Write(decodedText);

            if (StopConditionEvaluator.ShouldStop(decodedText, generated.Count))
            {
                break;
            }
        }
        Console.WriteLine();

        return tokenizer.Decode(generated.ToArray());
    }

    private void ApplyRepetitionPenalty(float[] probs, List<int> generated)
    {
        foreach (int token in generated)
        {
            if (token >= 0 && token < probs.Length)
            {
                probs[token] *= 0.8f;
            }
        }
    }
}