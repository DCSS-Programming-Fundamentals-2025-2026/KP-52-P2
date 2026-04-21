using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers
{
    public class FeedForwardLayer
    {
        private float[]? _lastInput;
        private float[]? _lastHidden;

        public float[] Compute(float[] x, TinyTransformerWeights weights, int d)
        {
            _lastInput = (float[])x.Clone();

            int dff = 4 * d;
            float[] hidden = new float[dff];

            for (int j = 0; j < dff; j++)
            {
                float sum = weights.Ffn1Bias[j];
                for (int i = 0; i < d; i++)
                {
                    sum += x[i] * weights.Ffn1[i, j];
                }
                hidden[j] = Math.Max(0, sum);
            }

            _lastHidden = (float[])hidden.Clone();

            float[] output = new float[d];
            for (int j = 0; j < d; j++)
            {
                float sum = weights.Ffn2Bias[j];
                for (int i = 0; i < dff; i++)
                {
                    sum += hidden[i] * weights.Ffn2[i, j];
                }
                output[j] = sum;
            }

            return output;
        }

        public float[] Backward(float[] dOutput, TinyTransformerWeights weights, float lr)
        {
            int d = _lastInput!.Length;
            int dff = 4 * d;

            float[] dHidden = new float[dff];

            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < dff; i++)
                {
                    weights.Ffn2[i, j] -= lr * _lastHidden![i] * dOutput[j];
                    dHidden[i] += dOutput[j] * weights.Ffn2[i, j];
                }
                weights.Ffn2Bias[j] -= lr * dOutput[j];
            }

            float[] dPreActivation = new float[dff];
            for (int i = 0; i < dff; i++)
            {
                dPreActivation[i] = _lastHidden![i] > 0 ? dHidden[i] : 0f;
            }

            float[] dInput = new float[d];

            for (int j = 0; j < dff; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    weights.Ffn1[i, j] -= lr * _lastInput[i] * dPreActivation[j];
                    dInput[i] += dPreActivation[j] * weights.Ffn1[i, j];
                }
                weights.Ffn1Bias[j] -= lr * dPreActivation[j];
            }

            return dInput;
        }
    }
}
