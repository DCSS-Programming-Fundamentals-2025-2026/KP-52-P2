using System.Globalization;

public static class CommandLineParser
{
    public static ReplOptions Parse(string[] args)
    {
        ReplOptions options = new ReplOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i].ToLower().Trim();

            if (arg == "--chat") continue;

            if (arg == "--checkpoint" && i + 1 < args.Length)
            {
                options.CheckpointPath = args[i + 1];
                Console.WriteLine(options.CheckpointPath);
                i++;
            }
            else if (arg == "--temp" && i + 1 < args.Length)
            {
                // Використовуємо InvariantCulture, щоб завжди працювала крапка (0.5)
                if (float.TryParse(args[i + 1], NumberStyles.Any, CultureInfo.InvariantCulture, out float parsedTemp))
                {
                    options.Temperature = parsedTemp;
                }
                i++;
            }
            else if (arg == "--topk" && i + 1 < args.Length)
            {
                if (int.TryParse(args[i + 1], out int parsedTopK))
                {
                    options.TopK = parsedTopK;
                }
                i++;
            }
            else if (arg == "--seed" && i + 1 < args.Length)
            {
                if (int.TryParse(args[i + 1], out int parsedSeed))
                {
                    options.Seed = parsedSeed;
                }
                i++;
            }
        }
        return options;
    }
}