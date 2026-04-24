using Contracts;
using Chat;
using Chat.TextGeneration;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Streams;
using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.NGram.Factories;
using Lib.Tokenization.Application;
using Lib.Training;
using Lib.Training.Configuration;           
using Lib.Sampling;
using System.Text.Json;
using Lib.Training.Metrics;

namespace MiniChatGPT.App
{
    public class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                PrintGeneralUsage();

                Console.WriteLine();
                Console.WriteLine();

                PrintTrainUsage();
                return;
            }

            ReplOptions options = CommandLineParser.Parse(args);
            string command = args[0].ToLowerInvariant();

            if (command == "chat" || command == "--chat")
            {
                RunChatMode(options);
            }
            else if (command == "train" || command == "--train")
            {
                RunTrainMode(options);
            }
        }

        static void RunChatMode(ReplOptions options)
        {
            try
            {
                var pipeline = new ChatPipeline();
                pipeline.InitializeCheckpoint(options.CheckpointPath);

                ISampler sampler = new Sampler();
                ITextGenerator generator = new ModelTextGenerator(pipeline.Model, pipeline.Tokenizer, sampler);

                var repl = new InteractiveRepl(generator, options.MaxTokens);
                repl.Run(options.Temperature, options.TopK, options.Seed);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Chat Error]: {ex.Message}");
            }
        }

        static void RunTrainMode(ReplOptions options)
        {
            Console.WriteLine($"[Trainer] Training {options.ModelKind} on {options.DataPath}...");

            try
            {
                CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
                Corpus corpus = loader.Load(options.DataPath, new CorpusLoadOptions(Lowercase: true));

                ITokenizer? tokenizer = null;
                ILanguageModel? model = null;

                string effectiveModelKind = "";
                string effectiveTokenizerKind = "";

                if (!string.IsNullOrWhiteSpace(options.OutputPath) && File.Exists(options.OutputPath))
                {
                    try
                    {
                        Console.WriteLine($"[Trainer] Checkpoint found. Resuming from {options.OutputPath}...");

                        Checkpoint checkpoint = JsonCheckpointIO.Load(options.OutputPath);

                        effectiveModelKind = checkpoint.ModelKind.ToLowerInvariant();
                        effectiveTokenizerKind = checkpoint.TokenizerKind.ToLowerInvariant();

                        if (!string.Equals(options.ModelKind, effectiveModelKind, StringComparison.OrdinalIgnoreCase))
                        {
                            throw new InvalidDataException(
                                $"Checkpoint model kind '{checkpoint.ModelKind}' does not match requested '{options.ModelKind}'.");
                        }

                        if (!string.Equals(options.TokenizerKind, effectiveTokenizerKind, StringComparison.OrdinalIgnoreCase))
                        {
                            throw new InvalidDataException(
                                $"Checkpoint tokenizer kind '{checkpoint.TokenizerKind}' does not match requested '{options.TokenizerKind}'.");
                        }

                        tokenizer = RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);
                        model = RestoreModel(checkpoint.ModelKind, checkpoint.ModelPayload);

                        if (model.VocabSize != tokenizer.VocabSize)
                        {
                            throw new InvalidDataException(
                                $"Vocab mismatch: model vocabSize={model.VocabSize}, tokenizer vocabSize={tokenizer.VocabSize}.");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("[Trainer] No checkpoint found. Starting new training...");

                        effectiveModelKind = options.ModelKind.ToLowerInvariant();
                        effectiveTokenizerKind = options.TokenizerKind.ToLowerInvariant();

                        tokenizer = effectiveTokenizerKind == "char"
                            ? CharTokenizer.BuildFromText(corpus.TrainText)
                            : WordTokenizer.BuildFromText(corpus.TrainText);

                        model = CreateNewModel(effectiveModelKind, tokenizer.VocabSize, options.Seed);
                    }
                }
                else
                {
                    Console.WriteLine("[Trainer] No checkpoint found. Starting new training...");

                    effectiveModelKind = options.ModelKind.ToLowerInvariant();
                    effectiveTokenizerKind = options.TokenizerKind.ToLowerInvariant();

                    tokenizer = effectiveTokenizerKind == "char"
                        ? CharTokenizer.BuildFromText(corpus.TrainText)
                        : WordTokenizer.BuildFromText(corpus.TrainText);

                    model = CreateNewModel(effectiveModelKind, tokenizer.VocabSize, options.Seed);
                }

                int[] tokens = tokenizer.Encode(corpus.TrainText);

                TrainingLoop trainingLoop = new TrainingLoop();
                TrainingConfig tConfig = new TrainingConfig(options.Epochs, options.LearningRate, options.CheckpointInterval);
                BatchConfig bConfig = new BatchConfig(options.BatchSize, options.BlockSize);

                ITokenStream stream = new ArrayTokenStream(tokens);
                IBatchProvider batchProvider = new TokenBatchProvider(stream, new BatchWindowSampler());

                TrainingMetrics metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, options.OutputPath, effectiveTokenizerKind, tokenizer, options.Seed);

                Console.WriteLine($"Average loss: {metrics.AverageLoss}");
                Console.WriteLine($"Elapsed time: {metrics.ElapsedTime}");
                Console.WriteLine($"Total steps: {metrics.TotalSteps}");
                Console.WriteLine($"NGram count: {metrics.NGramCount}");
                Console.WriteLine($"Perpelity: {metrics.Perplexity}");

            }                                                                     
            catch (Exception ex)
            {
                Console.WriteLine($"[Training Error]: {ex.Message}");
            }
        }

        public static ITokenizer RestoreTokenizer(string tokenizerKind, object payload)
        {
            ITokenizerFactory factory;

            if (tokenizerKind == "word")
            {
                factory = new WordTokenizerFactory();
            }
            else if (tokenizerKind == "char")
            {
                factory = new CharTokenizerFactory();
            }
            else
            {
                throw new ArgumentException("Unknown tokenizer type: " + tokenizerKind);
            }

            return factory.FromPayload(payload);
        }

        static ILanguageModel RestoreModel(string modelKind, object payload)
        {
            JsonElement jsonPayload = (JsonElement)payload;

            if (modelKind == "bigram" || modelKind == "trigram")
            {
                NGramModelFactory factory = new NGramModelFactory();
                return factory.CreateFromPayload(modelKind, jsonPayload);
            }
            else if (modelKind == "tinynn")
            {
                TinyNNModelFactory factory = new TinyNNModelFactory();
                return factory.CreateFromPayload(jsonPayload, modelKind);    
            }                                                                        
            else if (modelKind == "tinytransformer")
            {
                TinyTransformerModelFactory factory = new TinyTransformerModelFactory();
                return factory.CreateFromPayload(jsonPayload);
            }
            else
            {
                throw new ArgumentException("Unknown model type: " + modelKind);
            }
        }

        static ILanguageModel CreateNewModel(string modelKind, int vocabSize, int? seed)
        {
            return modelKind switch
            {
                "bigram" => new NGramModelFactory().Create("bigram", vocabSize),
                "trigram" => new NGramModelFactory().Create("trigram", vocabSize),
                "tinynn" => new TinyNNModelFactory().CreateNewModel("tinynn", vocabSize),
                "tinytransformer" => new TinyTransformerModelFactory().Create(vocabSize, seed ?? 42),
                _ => throw new ArgumentException($"Unknown model type: {modelKind}")
            };
        }

        static void PrintGeneralUsage()
        {
            Console.WriteLine("\nMini-ChatGPT CLI");
            Console.WriteLine("Використання:");
            Console.WriteLine("  dotnet run -- chat [options]  - Запустити режим чату");
            Console.WriteLine("  dotnet run -- train [options] - Запустити режим навчання");
            Console.WriteLine("\nСпробуйте 'dotnet run -- train -help' для параметрів навчання.");
        }

        static void PrintTrainUsage()
        {
            Console.WriteLine("\nПараметри навчання (--train):");
            Console.WriteLine("  --data <path>      Шлях до тексту (default: ../data/sample.txt)");
            Console.WriteLine("  --model <type>     Тип: bigram, trigram, tinynn, tinytransformer");
            Console.WriteLine("  --tokenizer <type> Тип: word, char");
            Console.WriteLine("  --epochs <int>     Кількість епох");
            Console.WriteLine("  --lr <float>       Learning rate");
            Console.WriteLine("  --batch <size>     Кількість батчів для тренування");
            Console.WriteLine("  --block <size>     Розмір кількість токенів, що враховуватимуться під час тренування");
            Console.WriteLine("  --interval <value>      Інтервал збереження чекпоінтуіі");
        }
    }
}