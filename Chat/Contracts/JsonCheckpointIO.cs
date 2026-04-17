using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Contracts
{
    public static class JsonCheckpointIO
    {
        private static readonly JsonSerializerOptions SerializerOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        };

        public static Checkpoint Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("Path to checkpoint cannot be empty.", nameof(path));
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Checkpoint file was not found.", path);
            }

            string json = File.ReadAllText(path);

            if (string.IsNullOrWhiteSpace(json))
            {
                throw new InvalidDataException("Checkpoint file is empty.");
            }

            Checkpoint checkpoint = JsonSerializer.Deserialize<Checkpoint>(json, SerializerOptions);

            if (checkpoint == null)
            {
                throw new InvalidDataException("Checkpoint JSON could not be deserialized.");
            }

            Validate(checkpoint);
            return checkpoint;
        }

        public static void Save(string path, Checkpoint checkpoint)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("Path to checkpoint cannot be empty.", nameof(path));
            }

            if (checkpoint == null)
            {
                throw new ArgumentNullException(nameof(checkpoint));
            }

            Validate(checkpoint);

            string directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            string json = JsonSerializer.Serialize(checkpoint, SerializerOptions);
            File.WriteAllText(path, json);
        }

        private static void Validate(Checkpoint checkpoint)
        {
            if (string.IsNullOrWhiteSpace(checkpoint.ModelKind))
            {
                throw new InvalidDataException("Checkpoint ModelKind is missing.");
            }

            if (string.IsNullOrWhiteSpace(checkpoint.TokenizerKind))
            {
                throw new InvalidDataException("Checkpoint TokenizerKind is missing.");
            }

            if (checkpoint.TokenizerPayload == null)
            {
                throw new InvalidDataException("Checkpoint TokenizerPayload is missing.");
            }

            if (checkpoint.ModelPayload == null)
            {
                throw new InvalidDataException("Checkpoint ModelPayload is missing.");
            }

            if (string.IsNullOrWhiteSpace(checkpoint.ContractFingerprintChain))
            {
                throw new InvalidDataException("Checkpoint ContractFingerprintChain is missing.");
            }
        }
    }
}