using System;
using System.IO;
using System.Text.Json;

namespace Contracts
{
    public static class JsonCheckpointIO
    {
        public static Checkpoint Load(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException("Файл чекпоінту не знайдено за шляхом: " + filePath);
            }

            string jsonContent = File.ReadAllText(filePath);

            JsonSerializerOptions options = new JsonSerializerOptions();
            options.PropertyNameCaseInsensitive = true;

            Checkpoint checkpoint = JsonSerializer.Deserialize<Checkpoint>(jsonContent, options);

            if (checkpoint == null)
            {
                throw new InvalidOperationException("Не вдалося розпарсити чекпоінт.");
            }

            return checkpoint;
        }
    }
}