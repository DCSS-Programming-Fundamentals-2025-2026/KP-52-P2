using Contracts;
using Lib.Models.NGram;
using Lib.Models.Trigram;
public class NGramModelFactoryTests
{
    [Test]
    public void Create_Bigram()
    {
        // Arrange
        NGramModelFactory factory = new NGramModelFactory();

        // Act
        ILanguageModel actualModel = factory.Create("bigram", 4);

        // Assert
        Assert.That(actualModel, Is.Not.Null);
        Assert.That(actualModel, Is.InstanceOf<NGramModel>());

        Assert.That(((NGramModel)actualModel).VocabSize, Is.EqualTo(4));
    }

    [Test]
    public void Create_Trigram()
    {
        // Arrange
        NGramModelFactory factory = new NGramModelFactory();

        // Act
        ILanguageModel actualModel = factory.Create("trigram", 4);

        // Assert
        Assert.That(actualModel, Is.Not.Null);
        Assert.That(actualModel, Is.InstanceOf<TrigramModel>());
        Assert.That(((TrigramModel)actualModel).VocabSize, Is.EqualTo(4));
    }

    [Test]
    public void Create_Unknown()
    {
        // Arrange
        NGramModelFactory factory = new NGramModelFactory();

        // Act + Assert
        Assert.Throws<ArgumentException>(() => factory.Create("gram", 3));
    }
}

