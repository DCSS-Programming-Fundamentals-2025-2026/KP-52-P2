namespace Tests;

using Lib.Sampling;
using Lib.Sampling.Processing;
using NUnit.Framework;

[TestFixture]
public class SamplerTests
{
    [Test]
    public void Test1()
    {
        ISampler sampler = new Sampler();
        float[] probs = { 0.1f, 0.5f, 0.3f, 0.05f, 0.05f };
        float temp = 1.0f;
        int topK = 5;
        int seed = 55;

        int[] resultsRun1 = new int[10];
        int[] resultsRun2 = new int[10];
        
        Random rng1 = new Random(seed);
        for (int i = 0; i < 10; i++)
        {
            resultsRun1[i] = sampler.Sample(probs, temp, topK, rng1);
        }
        
        Random rng2 = new Random(seed);
        for (int i = 0; i < 10; i++)
        {
            resultsRun2[i] = sampler.Sample(probs, temp, topK, rng2);
        }
        
        for (int i = 0; i < 10; i++)
        {
            Assert.That(resultsRun1[i], Is.EqualTo(resultsRun2[i]), "Determinism is violated: the results vary.");
        }
    }
    
    [Test]
    public void Test2()
    {
        ISampler sampler = new Sampler();

        float[] probs = { 0.05f, 0.1f, 0.8f, 0.05f }; 
        float temp = 1.0f;
        int topK = 1;
        Random rng = new Random(123);

        for (int i = 0; i < 50; i++)
        {
            int result = sampler.Sample(probs, temp, topK, rng);
            
            Assert.That(result, Is.EqualTo(2), "Top-K clipping is not working properly.");
        }
    }
    
    [Test]
    public void Test3()
    {
        ISampler sampler = new Sampler();
        
        float[] probs = { 0.0f, 1.0f, 0.0f }; 
        
        float temp = 1.0f; 
        int topK = 3;
        int seed = 42; 
        
        int result = sampler.SampleWithSeed(probs, temp, topK, seed);
        
        Assert.That(result, Is.EqualTo(1), "At normal temperatures, the algorithm should correctly handle a standard probability distribution.");
    }
    
    [Test]
    public void Test4()
    {
        ISampler sampler = new Sampler();
        float[] probs = { 0.2f, 0.8f };
        float temp = 1.0f;
        int hugeTopK = 100;
        int seed = 100;
        
        int result = sampler.SampleWithSeed(probs, temp, hugeTopK, seed);
        
        Assert.That(result, Is.GreaterThanOrEqualTo(0).And.LessThan(2), "The algorithm must work correctly even if TopK exceeds the size of the array.");
    }
    
    [Test]
    public void Test5()
    {
        ISampler sampler = new Sampler();
        float[] probs = { 1.0f };
        float temp = 0.5f; 
        int topK = 5;
        int seed = 99;
        
        int result = sampler.SampleWithSeed(probs, temp, topK, seed);
        
        Assert.That(result, Is.EqualTo(0), "If the input array has only one element, the only possible index is 0.");
    }
    
    [Test]
    public void Test6()
    {
        ISampler sampler = new Sampler();
        float[] probs = { 0.4f, 0.6f };
        float temp = 1.0f;
        int topK = 2;
        
        int result = sampler.Sample(probs, temp, topK, (Random?)null);
        
        Assert.That(result, Is.InRange(0, 1), 
            "If `null` is passed instead of `Random`, the algorithm should generate it on its own and function properly.");
    }



    [Test]
    public void Test7()
    {
        float[] probs = { 0.4f, 0.6f };

        float[] lowTemp = TemperatureScaler.Scale(probs, 0.5f);
        float[] highTemp = TemperatureScaler.Scale(probs, 2.0f);

        float[] normLow = ProbabilityNormalizer.Normalize(lowTemp);
        float[] normHigh = ProbabilityNormalizer.Normalize(highTemp);

        Assert.That(normLow[1], Is.GreaterThan(normHigh[1]));
    }

    [Test]
    public void Test8()
    {
        ISampler sampler = new Sampler();
        
        float[] probs = { 0.1f, 0.7f, 0.2f }; 
        
        float temp = 0.01f; 
        int topK = 3;
        int seed = 42; 
        
        int result = sampler.SampleWithSeed(probs, temp, topK, seed);
        
        Assert.That(result, Is.EqualTo(1));
    }

    [Test]
    public void Test9()
    {
        ISampler sampler = new Sampler();
        
        float[] probs = { 0.1f, 0.7f, 0.2f }; 
        
        float temp = 1.0f; 
        int topK = 1;
        int seed = 42; 
        
        int result = sampler.SampleWithSeed(probs, temp, topK, seed);
        
        Assert.That(result, Is.EqualTo(1));
    }

    [Test]
    public void Test10()
    {
        NucleusSampler sampler = new NucleusSampler();
        
        float[] probs = { 0.1f, 0.7f, 0.2f }; 
        
        float temp = 0.01f; 
        float topP = 0.5f;
        int seed = 42; 
        
        int result = sampler.SampleWithSeed(probs, temp, topP, seed);
        
        Assert.That(result, Is.EqualTo(1));
    }

    [Test]
    public void Test11()
    {
        NucleusSampler sampler = new NucleusSampler();
        
        float[] probs = { 0.1f, 0.7f, 0.2f }; 
        
        float temp = 1.0f; 
        float topP = 1.0f;
        int seed = 42; 
        
        int result = sampler.SampleWithSeed(probs, temp, topP, seed);
        
        Assert.That(result, Is.EqualTo(1));
    }
}