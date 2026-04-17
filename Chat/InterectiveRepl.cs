using System;
using System.Globalization;
using Contracts;

namespace Chat
{
    public class InteractiveRepl
    {
        private readonly ITextGenerator _generator;
        private int _maxTokens;

        public InteractiveRepl(ITextGenerator generator, int maxTokens = 50)
        {
            _generator = generator;
            _maxTokens = maxTokens;
        }

        public void Run(float temperature, int topK, int? seed)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== Mini-ChatGPT REPL ===");
            Console.WriteLine("/help — список команд, /quit — вихід\n");

            while (true)
            {
                Console.Write("\nКористувач> ");
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
                        Console.WriteLine("[Система] Контекст скинуто.");
                        continue;
                    }

                    if (cmd == "/temp" && parts.Length > 1)
                    {
                        if (float.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out float t))
                        {
                            temperature = t;
                            Console.WriteLine($"[Система] Температура = {temperature}");
                        }
                        continue;
                    }

                    if (cmd == "/topk" && parts.Length > 1)
                    {
                        if (int.TryParse(parts[1], out int k))
                        {
                            topK = k;
                            Console.WriteLine($"[Система] TopK = {topK}");
                        }
                        continue;
                    }

                    if (cmd == "/seed")
                    {
                        if (parts.Length > 1 && int.TryParse(parts[1], out int s)) seed = s;
                        else seed = null;
                        Console.WriteLine($"[Система] Seed = {(seed.HasValue ? seed.Value.ToString() : "null")}");
                        continue;
                    }

                    if (cmd == "/maxtokens" && parts.Length > 1)
                    {
                        if (int.TryParse(parts[1], out int mt))
                        {
                            _maxTokens = mt;
                            Console.WriteLine($"[Система] MaxTokens = {_maxTokens}");
                        }
                        continue;
                    }

                    Console.WriteLine("[Система] Невідома команда. Введіть /help.");
                    continue;
                }

                try
                {
                    string response = _generator.Generate(input, _maxTokens, temperature, topK, seed);
                    Console.WriteLine($"Асистент: {response}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Помилка]: {ex.Message}");
                }
            }
        }

        private void ShowHelp()
        {
            Console.WriteLine("\n/help, /reset, /temp T, /topk K, /seed N, /maxtokens M, /quit\n");
        }
    }
}