using System;
using System.Globalization;
using Contracts;
using Lib.Sampling;
using Lib.Corpus;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Configuration;
using Lib.Tokenization.Application;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;
using Lib.Models.TinyTransformer.Factories;
using Chat.TextGeneration;
using Lib.Tokenization.Domain.Model;

namespace Chat.App
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                PrintUsage();
                return;
            }

            string command = args[0].ToLowerInvariant();

            if (command == "chat")
            {
                RunChat(args);
            }
            else if (command == "train")
            {
                RunTrain(args);
            }
            else
            {
                Console.WriteLine("Невідома команда: " + command);
                PrintUsage();
            }
        }

        static void RunChat(string[] args)
        {
            string checkpointPath = "checkpoint.json";
            float temperature = 0.3f;
            int topK = 5;
            int? seed = null;
            int maxTokens = 50;

            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "-checkpoint" && i + 1 < args.Length) { checkpointPath = args[++i]; }
                else if (args[i] == "-temp" && i + 1 < args.Length) { float.TryParse(args[++i], NumberStyles.Any, CultureInfo.InvariantCulture, out temperature); }
                else if (args[i] == "-topk" && i + 1 < args.Length) { int.TryParse(args[++i], out topK); }
                else if (args[i] == "-seed" && i + 1 < args.Length) { int.TryParse(args[++i], out int s); seed = s; }
                else if (args[i] == "-maxtokens" && i + 1 < args.Length) { int.TryParse(args[++i], out maxTokens); }
            }

            var pipeline = new ChatPipeline();
            pipeline.InitializeCheckpoint(checkpointPath);

            ISampler sampler = new Sampler();
            ITextGenerator generator = new ModelTextGenerator(pipeline.Model, pipeline.Tokenizer, sampler);

            var repl = new InteractiveRepl(generator, maxTokens);
            repl.Run(temperature, topK, seed);
        }

        static void RunTrain(string[] args)
        {
            string corpusPath = "";
            int epochs = 10;
            float lr = 0.01f;
            int embeddingSize = 32;
            int contextSize = 16;
            int? seed = null;
            string outputPath = "checkpoint.json";
            string tokenizerKind = "word";

            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "-corpus" && i + 1 < args.Length) { corpusPath = args[++i]; }
                else if (args[i] == "-epochs" && i + 1 < args.Length) { int.TryParse(args[++i], out epochs); }
                else if (args[i] == "-lr" && i + 1 < args.Length) { float.TryParse(args[++i], NumberStyles.Any, CultureInfo.InvariantCulture, out lr); }
                else if (args[i] == "-embed" && i + 1 < args.Length) { int.TryParse(args[++i], out embeddingSize); }
                else if (args[i] == "-context" && i + 1 < args.Length) { int.TryParse(args[++i], out contextSize); }
                else if (args[i] == "-seed" && i + 1 < args.Length) { int.TryParse(args[++i], out int s); seed = s; }
                else if (args[i] == "-output" && i + 1 < args.Length) { outputPath = args[++i]; }
                else if (args[i] == "-tokenizer" && i + 1 < args.Length) { tokenizerKind = args[++i].ToLowerInvariant(); }
            }

            if (string.IsNullOrEmpty(corpusPath))
            {
                Console.WriteLine("Помилка: необхідно вказати шлях до корпусу (-corpus <шлях>).");
                return;
            }

            Console.WriteLine($"Навчання TinyTransformer на корпусі: {corpusPath}");
            Console.WriteLine($"  EmbeddingSize={embeddingSize}, ContextSize={contextSize}, Epochs={epochs}, LR={lr}, Tokenizer={tokenizerKind}");

            var loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load(corpusPath, new CorpusLoadOptions { Lowercase = true, ValidationFraction = 0.1 });

            Console.WriteLine($"  Корпус: train={corpus.TrainText.Length} символів, val={corpus.ValText.Length} символів");

            ITokenizer tokenizer;
            if (tokenizerKind == "word")
            {
                tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            }
            else
            {
                tokenizer = CharTokenizer.BuildFromText(corpus.TrainText);
            }
            int[] tokens = tokenizer.Encode(corpus.TrainText);

            Console.WriteLine($"  Словник: {tokenizer.VocabSize} символів, токенів: {tokens.Length}");

            var factory = new TinyTransformerModelFactory();
            var model = seed.HasValue
                ? factory.Create(tokenizer.VocabSize, embeddingSize, 1, contextSize, seed.Value)
                : factory.Create(tokenizer.VocabSize, embeddingSize, 1, contextSize);

            Console.WriteLine("Початок навчання...");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float totalLoss = 0f;
                int count = 0;

                for (int i = contextSize; i < tokens.Length; i++)
                {
                    int[] context = new int[contextSize];
                    for (int j = 0; j < contextSize; j++)
                    {
                        context[j] = tokens[i - contextSize + j];
                    }

                    float loss = model.TrainStep(context, tokens[i], lr);

                    if (float.IsNaN(loss) || float.IsInfinity(loss))
                    {
                        Console.WriteLine($"  Попередження: loss={loss} на кроці {count}, зупинка епохи");
                        break;
                    }

                    totalLoss += loss;
                    count++;

                    if (count % 10000 == 0)
                    {
                        Console.WriteLine($"  Епоха {epoch + 1}: оброблено {count}/{tokens.Length - contextSize} кроків, поточний loss={loss:F4}");
                    }
                }

                float avgLoss = totalLoss / count;
                Console.WriteLine($"Епоха {epoch + 1}/{epochs}: avg loss = {avgLoss:F4}");
            }

            Console.WriteLine("Збереження чекпоінту...");

            string fingerprint = "V1_tinytransformer:vocabSize=" + tokenizer.VocabSize;

            var checkpoint = new Checkpoint(
                ModelKind: "tinytransformer",
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed ?? 0,
                ContractFingerprintChain: fingerprint
            );

            JsonCheckpointIO.Save(outputPath, checkpoint);
            Console.WriteLine($"Чекпоінт збережено: {outputPath}");
        }

        static void PrintUsage()
        {
            Console.WriteLine("Використання:");
            Console.WriteLine("  dotnet run -- chat -checkpoint <шлях> [-temp 0.3] [-topk 5] [-seed 42] [-maxtokens 50]");
            Console.WriteLine("  dotnet run -- train -corpus <шлях> [-epochs 10] [-lr 0.01] [-embed 32] [-context 16] [-seed 42] [-output checkpoint.json] [-tokenizer word|char]");
        }
    }
}
