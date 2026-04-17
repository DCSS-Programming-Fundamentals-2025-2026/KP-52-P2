using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.Layers;

public class SelfAttentionLayerTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;
    private TinyTransformerWeights _weights;
    private SelfAttentionLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        _layer = new SelfAttentionLayer();
    }

    [Test]
    public void Backward_ReturnsCorrectShape()
    {
        int seqLen = 3;
        float[][] input = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = 0.1f * (i + j);
            }
        }

        _layer.Compute(input, _weights, _embeddingSize);

        float[] dFinalAttnOut = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dFinalAttnOut[i] = 0.01f;

        float[][] dX = _layer.Backward(dFinalAttnOut, _weights, 0.01f);

        Assert.That(dX.Length, Is.EqualTo(seqLen));
        for (int i = 0; i < seqLen; i++)
        {
            Assert.That(dX[i].Length, Is.EqualTo(_embeddingSize));
        }
    }

    [Test]
    public void Backward_UpdatesAttentionWeights()
    {
        int seqLen = 4;
        float[][] input = new float[seqLen][];
        var rng = new Random(123);
        for (int i = 0; i < seqLen; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _layer.Compute(input, _weights, _embeddingSize);

        var wqSnapshot = (float[,])_weights.Wq.Clone();
        var wkSnapshot = (float[,])_weights.Wk.Clone();
        var wvSnapshot = (float[,])_weights.Wv.Clone();
        var woSnapshot = (float[,])_weights.Wo.Clone();

        float[] dFinalAttnOut = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dFinalAttnOut[i] = 0.1f;

        _layer.Backward(dFinalAttnOut, _weights, 0.01f);

        bool wqChanged = false, wkChanged = false, wvChanged = false, woChanged = false;
        for (int i = 0; i < _embeddingSize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                if (_weights.Wq[i, j] != wqSnapshot[i, j]) wqChanged = true;
                if (_weights.Wk[i, j] != wkSnapshot[i, j]) wkChanged = true;
                if (_weights.Wv[i, j] != wvSnapshot[i, j]) wvChanged = true;
                if (_weights.Wo[i, j] != woSnapshot[i, j]) woChanged = true;
            }
        }

        Assert.That(wqChanged, Is.True);
        Assert.That(wkChanged, Is.True);
        Assert.That(wvChanged, Is.True);
        Assert.That(woChanged, Is.True);
    }

    [Test]
    public void Backward_DoesNotReturnAllZeros()
    {
        float[][] input = new float[2][];
        for (int i = 0; i < 2; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = 1.0f;
            }
        }

        _layer.Compute(input, _weights, _embeddingSize);

        float[] dFinalAttnOut = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dFinalAttnOut[i] = 1.0f;

        float[][] dX = _layer.Backward(dFinalAttnOut, _weights, 0.01f);

        bool hasNonZero = dX.Any(row => row.Any(v => v != 0));
        Assert.That(hasNonZero, Is.True);
    }
}
