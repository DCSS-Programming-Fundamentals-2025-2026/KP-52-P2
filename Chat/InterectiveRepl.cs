using System;
using System.Globalization;
using Contracts;

namespace Chat
{
    public class InteractiveRepl
    {
        private readonly ITextGenerator _generator;

        public InteractiveRepl(ITextGenerator generator)
        {
            _generator = generator;
        }

        public void Run(float temperature, int topK, int? seed)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== Mini-ChatGPT REPL ===");

            while (true)
            {
                Console.Write("\nUser: ");
                string input = Console.ReadLine() ?? string.Empty;

                if (string.IsNullOrWhiteSpace(input)) continue;

                if (input.StartsWith("/"))
                {
                    string[] parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    string cmd = parts[0].ToLowerInvariant();

                    if (cmd == "/quit") break;

                    if (cmd == "/help")
                    {
                        ShowHelp();
                        continue;
                    }

                    if (cmd == "/reset")
                    {
                        Console.WriteLine("[System] Context reset.");
                        continue;
                    }

                    if (cmd == "/temp" && parts.Length > 1)
                    {
                        if (float.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out float t))
                        {
                            temperature = t;
                            Console.WriteLine($"[System] Temp = {temperature}");
                        }
                        continue;
                    }

                    if (cmd == "/topk" && parts.Length > 1)
                    {
                        if (int.TryParse(parts[1], out int k))
                        {
                            topK = k;
                            Console.WriteLine($"[System] TopK = {topK}");
                        }
                        continue;
                    }

                    if (cmd == "/seed")
                    {
                        if (parts.Length > 1 && int.TryParse(parts[1], out int s)) seed = s;
                        else seed = null;
                        Console.WriteLine($"[System] Seed = {(seed.HasValue ? seed.Value.ToString() : "null")}");
                        continue;
                    }

                    Console.WriteLine("[System] Unknown command.");
                    continue;
                }

                try
                {
                    string response = _generator.Generate(input, temperature, topK, seed);
                    Console.WriteLine($"Assistant: {response}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Error]: {ex.Message}");
                }
            }
        }

        private void ShowHelp()
        {
            Console.WriteLine("\n/help, /reset, /temp T, /topk K, /seed N, /quit\n");
        }
    }
}