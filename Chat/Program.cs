using System;
using Contracts;
using Lib.Sampling;

namespace Chat.App
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Chat System...");

            var options = CommandLineParser.Parse(args);

            ISampler sampler = new Sampler();
            ITextGenerator model = new IntegratedMockModel(sampler);

            var chat = new InteractiveRepl(model);

            chat.Run(options.Temperature, options.TopK, options.Seed);
        }
    }
}