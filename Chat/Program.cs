using System;
using Contracts;
using Chat;
using Chat.TextGeneration;
using Lib.Sampling;

class Program
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("Консольне тестування системи Mini-ChatGPT.");

        ReplOptions options = CommandLineParser.Parse(args);

        ITextGenerator generator;

        if (!string.IsNullOrEmpty(options.CheckpointPath))
        {
            try
            {

                ChatPipeline pipeline = new ChatPipeline();
                pipeline.InitializeCheckpoint(options.CheckpointPath);

                
                ISampler sampler = new Sampler();
                
                
                generator = new ModelTextGenerator(
                    pipeline.Model, 
                    pipeline.Tokenizer, 
                    sampler
                );
                
                Console.WriteLine($"[System] Model loaded from: {options.CheckpointPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Critical Error] Failed to load checkpoint: {ex.Message}");
                Console.WriteLine("[System] Falling back to Mock Model...");
                Console.WriteLine($"[Stack Trace]: {ex.StackTrace}");
                generator = new IntegratedMockModel(new Sampler());
            }
        }
        else
        {
            Console.WriteLine("[System] No checkpoint specified. Starting with IntegratedMockModel.");
            generator = new IntegratedMockModel(new Sampler());
        }


        ChatRepl chat = new ChatRepl(generator);
        
        
        chat.Run(options.Temperature, options.TopK, options.Seed);
    }
}