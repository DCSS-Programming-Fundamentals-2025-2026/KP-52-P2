using System.Text.Json;
using Lib.Tokenization.Application;
using Lib.Tokenization.Infrastructure;

namespace Lib.Tokenization.Tests.Infrastructure;

[TestFixture]
public class TokenizerFactoryTests
{
    [Test]
    public void BuildFromPayload_AfterSerialization_RestoresExactBehavior()
    {
        var original = WordTokenizer.BuildFromText("привіт світ привіт");
        var originalPayload = original.GetPayloadForCheckpoint();

        string json = JsonSerializer.Serialize(originalPayload);
    
        JsonElement restoredPayload = JsonDocument.Parse(json).RootElement;

        var factory = new WordTokenizerFactory(); 
        var restored = factory.FromPayload(restoredPayload);

        string testText = "світ привіт";
        Assert.That(restored.Encode(testText), Is.EqualTo(original.Encode(testText)));
        Assert.That(restored.VocabSize, Is.EqualTo(original.VocabSize));
        Assert.That(restored.Decode(restored.Encode(testText)), Is.EqualTo("світ привіт"));
    }
}