public static class CommandLineParser
{
    public static ReplOptions Parse(string[] args)
    {
        ReplOptions options = new ReplOptions();

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--chat")
            {
                continue;
            }
            else if (args[i] == "--checkpoint" && i + 1 < args.Length)
            {
                options.CheckpointPath = args[i + 1];
                i++;
            }
            else if (args[i] == "--temp" && i + 1 < args.Length)
            {
                float parsedTemp = 0f;
                if (float.TryParse(args[i + 1], out parsedTemp))
                {
                    options.Temperature = parsedTemp;
                }
                i++;
            }
            else if (args[i] == "--topk" && i + 1 < args.Length)
            {
                int parsedTopK = 0;
                if (int.TryParse(args[i + 1], out parsedTopK))
                {
                    options.TopK = parsedTopK;
                }
                i++;
            }
            else if (args[i] == "--seed" && i + 1 < args.Length)
            {
                int parsedSeed = 0;
                if (int.TryParse(args[i + 1], out parsedSeed))
                {
                    options.Seed = parsedSeed;
                }
                i++;
            }
        }
        return options;
    }
}