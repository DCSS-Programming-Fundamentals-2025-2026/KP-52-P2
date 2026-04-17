using System;
using System.Text.Json;
using Contracts;
using Lib.Tokenization.Application;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.NGram;
using Lib.Models.TinyTransformer;

namespace Chat
{
    public class ChatPipeline
    {
        public ITokenizer Tokenizer { get; private set; } = null!;
        public ILanguageModel Model { get; private set; } = null!;

        public void InitializeCheckpoint(string checkpointPath)
        {
            Console.WriteLine("Завантаження чекпоінту...");
            Checkpoint checkpoint = JsonCheckpointIO.Load(checkpointPath);

            Console.WriteLine("Відновлення токенізатора...");
            Tokenizer = RestoreTokenizer(checkpoint.TokenizerKind, checkpoint.TokenizerPayload);

            Console.WriteLine("Відновлення моделі...");
            Model = RestoreModel(checkpoint.ModelKind, Tokenizer.VocabSize, checkpoint.ModelPayload);

            Console.WriteLine("Перевірка FingerPrint...");
            VerifyFingerprint(checkpoint.ContractFingerprintChain, Model);

            Console.WriteLine("Успіх! Модель готова до чату.");
        }

        private ITokenizer RestoreTokenizer(string tokenizerKind, object payload)
        {
            ITokenizerFactory factory;

            if (tokenizerKind == "word")
            {
                factory = new WordTokenizerFactory();
            }
            else if (tokenizerKind == "char")
            {
                factory = new CharTokenizerFactory(); 
            }
            else
            {
                throw new ArgumentException("Невідомий тип токенізатора: " + tokenizerKind);
            }

            return factory.FromPayload(payload);
        }

        private ILanguageModel RestoreModel(string modelKind, int vocabSize, object payload)
        {
            JsonElement jsonPayload = (JsonElement)payload;

            if (modelKind == "bigram" || modelKind == "trigram")
            {
                NGramModelFactory factory = new NGramModelFactory();
                return factory.CreateFromPayload(modelKind, jsonPayload);
            }
            else if (modelKind == "tinynn")
            {
                TinyNNModelFactory factory = new TinyNNModelFactory();
                return factory.CreateFromPayload(jsonPayload, modelKind);
            }
            else if (modelKind == "tinytransformer")
            {
                TinyTransformerModelFactory factory = new TinyTransformerModelFactory();
                return factory.CreateFromPayload(jsonPayload);
            }
            else
            {
                throw new ArgumentException("Невідомий тип моделі: " + modelKind);
            }
        }

        private void VerifyFingerprint(string expectedFingerprint, ILanguageModel model)
        {
            string actualFingerprint = "V1_" + model.ModelKind + ":vocabSize=" + model.VocabSize;

            if (string.IsNullOrEmpty(expectedFingerprint))
            {
                Console.WriteLine("Попередження: У чекпоінті відсутній FingerPrint для перевірки.");
                return;
            }

            if (!expectedFingerprint.Contains(actualFingerprint))
            {
                throw new Exception("Помилка FingerPrint! Чекпоінт не сумісний з поточним кодом.\n" +
                                    "Очікувалося (у файлі): " + expectedFingerprint + "\n" +
                                    "Отримано (від моделі): " + actualFingerprint);
            }
        }
    }
}