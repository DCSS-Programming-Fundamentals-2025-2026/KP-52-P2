using System.Globalization;

public static class CommandLineParser
{
    public static ReplOptions Parse(string[] args)
    {
        ReplOptions options = new ReplOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i].ToLower().Trim();

            if (arg == "--chat" || arg == "--train") continue;

            if (i + 1 >= args.Length) continue;
            string val = args[i + 1];

            switch (arg)
            {
                case "--checkpoint":
                    options.CheckpointPath = val;
                    i++;
                    break;
                case "--temp":
                    if (float.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out float t)) options.Temperature = t;
                    i++;
                    break;
                case "--topk":
                    if (int.TryParse(val, out int k)) options.TopK = k;
                    i++;
                    break;
                case "--seed":
                    if (int.TryParse(val, out int s)) options.Seed = s;
                    i++;
                    break;
                case "--maxtokens":
                    if (int.TryParse(val, out int mt)) options.MaxTokens = mt;
                    i++;
                    break;


                case "--data":
                    options.DataPath = val;
                    i++;
                    break;
                case "--model":
                    options.ModelKind = val.ToLower();
                    i++;
                    break;
                case "--tokenizer":
                    options.TokenizerKind = val.ToLower();
                    i++;
                    break;
                case "--epochs":
                    if (int.TryParse(val, out int ep)) options.Epochs = ep;
                    i++;
                    break;
                case "--out":
                    options.OutputPath = val;
                    i++;
                    break;
                case "--lr":
                    if (float.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out float lr)) options.LearningRate = lr;
                    i++;
                    break;
                case "--batch":
                    if (int.TryParse(val, out int bs)) options.BatchSize = bs;
                    i++;
                    break;
                case "--block":
                    if (int.TryParse(val, out int bl)) options.BlockSize = bl;
                    i++;
                    break;
                case "--interval":
                    if (int.TryParse(val, out int inter)) options.CheckpointInterval = inter;
                    i++;
                    break;
            }
        }
        return options;
    }
}