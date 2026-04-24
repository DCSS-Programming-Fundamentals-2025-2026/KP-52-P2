using System;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;

namespace Lib.Batching;

public class TokenBatchProvider : IBatchProvider
{
    private readonly ITokenStream _stream;
    private readonly BatchWindowSampler _sampler;

    public TokenBatchProvider(ITokenStream stream, BatchWindowSampler sampler)
    {
        _stream = stream;
        _sampler = sampler;
    }

    public Batch GetBatch(BatchConfig config, Random rng)
    {
        var tokens = _stream.GetTokens();

        int blockSize = config.BlockSize > tokens.Length ? tokens.Length - 1: config.BlockSize;
        if (tokens.Length < 2 || blockSize <= 0)
        {
            throw new ArgumentException("Створення батчу неможливе!");
        }

        var startIndices = _sampler.GetRandomStartIndices(tokens.Length, config.BatchSize, blockSize, rng);

        int[][] inputs = new int[config.BatchSize][];
        int[] targets = new int[config.BatchSize];

        for (int i = 0; i < config.BatchSize; i++)
        {
            int startIndex = startIndices[i];
            
            inputs[i] = new int[blockSize];
            Array.Copy(tokens, startIndex, inputs[i], 0, blockSize);
            
            targets[i] = tokens[startIndex + blockSize];
        }

        return new Batch(inputs, targets);
    }
}