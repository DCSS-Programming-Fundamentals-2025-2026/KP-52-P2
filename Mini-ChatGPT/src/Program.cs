using System;
using System.IO;
using System.Globalization;
using System.Linq;
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

namespace MiniChatGPT.App
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0) { PrintGeneralUsage(); return; }

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

                ITokenizer tokenizer = options.TokenizerKind == "char" 
                    ? CharTokenizer.BuildFromText(corpus.TrainText) 
                    : WordTokenizer.BuildFromText(corpus.TrainText);

                int[] tokens = tokenizer.Encode(corpus.TrainText);

                ILanguageModel model = options.ModelKind switch
                {
                    "bigram" => new NGramModelFactory().Create("bigram", tokenizer.VocabSize),
                    "trigram" => new NGramModelFactory().Create("trigram", tokenizer.VocabSize),
                    "tinynn" => new TinyNNModelFactory().CreateNewModel("tinynn", tokenizer.VocabSize),
                    "tinytransformer" => new TinyTransformerModelFactory().Create(tokenizer.VocabSize, options.Seed ?? 42),
                    _ => throw new ArgumentException($"Unknown model type: {options.ModelKind}")
                };

                TrainingLoop trainingLoop = new TrainingLoop();
                TrainingConfig tConfig = new TrainingConfig(options.Epochs, options.LearningRate, options.CheckpointInterval);
                BatchConfig bConfig = new BatchConfig(options.BatchSize, options.BlockSize);
                
                ITokenStream stream = new ArrayTokenStream(tokens);
                IBatchProvider batchProvider = new TokenBatchProvider(stream, new BatchWindowSampler());

                trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, options.OutputPath);

                
                var checkpoint = new Checkpoint(
                    ModelKind: options.ModelKind,
                    TokenizerKind: options.TokenizerKind,
                    TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                    ModelPayload: model.GetPayloadForCheckpoint(),
                    Seed: options.Seed ?? 42,
                    ContractFingerprintChain: $"V1_{options.ModelKind}:vocabSize={tokenizer.VocabSize}"
                );

                JsonCheckpointIO.Save(options.OutputPath, checkpoint);
                Console.WriteLine($"[Success] Model saved to {options.OutputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Training Error]: {ex.Message}");
            }
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
            Console.WriteLine("  --out <path>       Шлях для збереження чекпоінту");
        }
    }
}