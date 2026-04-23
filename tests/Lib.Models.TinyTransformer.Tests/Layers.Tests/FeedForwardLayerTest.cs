using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.Layers;

public class FeedForwardLayerTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;
    private TinyTransformerWeights _weights;
    private FeedForwardLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        _layer = new FeedForwardLayer();
    }

    [Test]
    public void Backward_ReturnsCorrectSize()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) input[i] = 0.1f * i;

        _layer.Compute(input, _weights, _embeddingSize);
        float[] dOutput = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dOutput[i] = 0.01f;

        float[] dInput = _layer.Backward(dOutput, _weights, 0.01f);

        Assert.That(dInput.Length, Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Backward_UpdatesWeights()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) input[i] = 1.0f;

        _layer.Compute(input, _weights, _embeddingSize);

        var ffn1Snapshot = (float[,])_weights.Ffn1.Clone();
        var ffn2Snapshot = (float[,])_weights.Ffn2.Clone();

        float[] dOutput = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dOutput[i] = 0.1f;

        _layer.Backward(dOutput, _weights, 0.01f);

        bool ffn1Changed = false, ffn2Changed = false;
        for (int i = 0; i < _embeddingSize; i++)
        {
            for (int j = 0; j < 4 * _embeddingSize; j++)
            {
                if (_weights.Ffn1[i, j] != ffn1Snapshot[i, j]) ffn1Changed = true;
            }
        }
        for (int i = 0; i < 4 * _embeddingSize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                if (_weights.Ffn2[i, j] != ffn2Snapshot[i, j]) ffn2Changed = true;
            }
        }

        Assert.That(ffn1Changed, Is.True);
        Assert.That(ffn2Changed, Is.True);
    }

    [Test]
    public void Backward_DoesNotReturnAllZeros()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) input[i] = 1.0f;

        _layer.Compute(input, _weights, _embeddingSize);

        float[] dOutput = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++) dOutput[i] = 1.0f;

        float[] dInput = _layer.Backward(dOutput, _weights, 0.01f);

        Assert.That(dInput.Any(v => v != 0), Is.True);
    }
}
