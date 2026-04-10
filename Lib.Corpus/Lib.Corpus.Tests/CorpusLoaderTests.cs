using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;

namespace Lib.Corpus.Tests;

public class FakeFileSystem : IFileSystem
{
    public bool FileExists { get; set; } = true;
    public string FileContent { get; set; } = string.Empty;

    public bool Exists(string path) => FileExists;

    public string ReadAllText(string path) => FileContent;
}

public class CorpusLoaderTests
{
    private FakeFileSystem _fileSystem;
    private CorpusLoader _loader;

    [SetUp]
    public void Setup()
    {
        _fileSystem = new FakeFileSystem();
        _loader = new CorpusLoader(_fileSystem);
    }

    [Test]
    public void LoadNormalPathSplitsTextCorrectly()
    {
        _fileSystem.FileExists = true;
        _fileSystem.FileContent = "1234567890";
        var options = new CorpusLoadOptions(ValidationFraction: 0.2);

        var result = _loader.Load("dummy.txt", options);

        Assert.That(result.TrainText, Is.EqualTo("12345678"));
        Assert.That(result.ValText, Is.EqualTo("90"));
    }

    [Test]
    public void LoadWithLowercaseOptionNormalizesText()
    {
        _fileSystem.FileExists = true;
        _fileSystem.FileContent = "HeLlO WoRlD";
        var options = new CorpusLoadOptions(Lowercase: true, ValidationFraction: 0.0);

        var result = _loader.Load("dummy.txt", options);

        Assert.That(result.TrainText, Is.EqualTo("hello world"));
    }

    [Test]
    public void LoadMissingFileUsesFallbackText()
    {
        _fileSystem.FileExists = false;
        var options = new CorpusLoadOptions(FallbackText: "fallback_data", ValidationFraction: 0.0);

        var result = _loader.Load("missing.txt", options);

        Assert.That(result.TrainText, Is.EqualTo("fallback_data"));
    }

    [Test]
    public void LoadFromTextEmptyText()
    {
        var options = new CorpusLoadOptions(ValidationFraction: 0.1);

        var result = _loader.LoadFromText(string.Empty, options);

        Assert.That(result.TrainText, Is.EqualTo(string.Empty));
        Assert.That(result.ValText, Is.EqualTo(string.Empty));
    }

    [Test]
    public void SplitNegativeValidationFraction()
    {
        var options = new CorpusLoadOptions(ValidationFraction: -0.1);

        Assert.Throws<ArgumentOutOfRangeException>(() => _loader.LoadFromText("1234567890", options));
    }

    [Test]
    public void SplitValidationFractionGreaterThanOne()
    {
        var options = new CorpusLoadOptions(ValidationFraction: 1.1);

        Assert.Throws<ArgumentOutOfRangeException>(() => _loader.LoadFromText("1234567890", options));
    }

    [Test]
    public void LoadFromTextNonZeroFraction()
    {
        var options = new CorpusLoadOptions(ValidationFraction: 0.3);

        var result = _loader.LoadFromText("1234567890", options);

        Assert.That(result.TrainText, Is.EqualTo("1234567"));
        Assert.That(result.ValText, Is.EqualTo("890"));
    }

    [Test]
    public void LoadFromTextShortTextRoundsToZero()
    {
        var options = new CorpusLoadOptions(ValidationFraction: 0.2);

        var result = _loader.LoadFromText("abc", options);

        Assert.That(result.TrainText, Is.EqualTo("abc"));
        Assert.That(result.ValText, Is.EqualTo(string.Empty));
    }
}