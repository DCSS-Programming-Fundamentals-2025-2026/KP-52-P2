using Lib.MathCore;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers
{
    public class SelfAttentionLayer
    {
        private float[][]? _lastInput;
        private float[][]? _lastQ;
        private float[][]? _lastK;
        private float[][]? _lastV;
        private float[][]? _lastAlpha;
        private float[][]? _lastRawAttnOut;

        public float[][] Compute(float[][] x, TinyTransformerWeights weights, int d)
        {
            _lastInput = DeepClone(x);

            int n = x.Length;
            float[][] Q = Multiply(x, weights.Wq, d);
            float[][] K = Multiply(x, weights.Wk, d);
            float[][] V = Multiply(x, weights.Wv, d);

            _lastQ = DeepClone(Q);
            _lastK = DeepClone(K);
            _lastV = DeepClone(V);

            float[][] scores = new float[n][];
            float scale = (float)Math.Sqrt(d);

            for (int i = 0; i < n; i++)
            {
                scores[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    if (j > i)
                    {
                        scores[i][j] = float.NegativeInfinity;
                        continue;
                    }

                    float dot = 0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += Q[i][k] * K[j][k];
                    }
                    scores[i][j] = Math.Clamp(dot / scale, -10f, 10f);
                }
            }

            float[][] alpha = new float[n][];
            float[][] attnOut = new float[n][];
            for (int i = 0; i < n; i++)
            {
                alpha[i] = MathOps.Default.Softmax(scores[i]);
                attnOut[i] = new float[d];
                for (int j = 0; j <= i; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        attnOut[i][k] += alpha[i][j] * V[j][k];
                    }
                }
            }

            _lastAlpha = DeepClone(alpha);
            _lastRawAttnOut = DeepClone(attnOut);

            return Multiply(attnOut, weights.Wo, d);
        }

        public float[][] Backward(float[] dFinalAttnOutLastPos, TinyTransformerWeights weights, float lr)
        {
            int n = _lastInput!.Length;
            int d = _lastInput[0].Length;
            int lastPos = n - 1;
            float scale = (float)Math.Sqrt(d);

            float[] dRawAttnOutLast = new float[d];
            for (int k = 0; k < d; k++)
            {
                for (int j = 0; j < d; j++)
                {
                    dRawAttnOutLast[k] += dFinalAttnOutLastPos[j] * weights.Wo[k, j];
                }
            }

            for (int k = 0; k < d; k++)
            {
                for (int j = 0; j < d; j++)
                {
                    weights.Wo[k, j] -= lr * _lastRawAttnOut![lastPos][k] * dFinalAttnOutLastPos[j];
                }
            }

            float[] dAlphaLast = new float[n];
            float[][] dV = new float[n][];
            for (int j = 0; j < n; j++)
            {
                dV[j] = new float[d];
            }

            for (int j = 0; j <= lastPos; j++)
            {
                for (int k = 0; k < d; k++)
                {
                    dAlphaLast[j] += dRawAttnOutLast[k] * _lastV![j][k];
                    dV[j][k] += _lastAlpha![lastPos][j] * dRawAttnOutLast[k];
                }
            }

            float[] dScoresLast = new float[n];
            float dotProduct = 0f;
            for (int j = 0; j < n; j++)
            {
                dotProduct += _lastAlpha![lastPos][j] * dAlphaLast[j];
            }
            for (int j = 0; j < n; j++)
            {
                dScoresLast[j] = _lastAlpha[lastPos][j] * (dAlphaLast[j] - dotProduct);
                dScoresLast[j] /= scale;
            }

            float[] dQLast = new float[d];
            float[][] dK = new float[n][];
            for (int j = 0; j < n; j++)
            {
                dK[j] = new float[d];
            }

            for (int j = 0; j <= lastPos; j++)
            {
                for (int k = 0; k < d; k++)
                {
                    dQLast[k] += dScoresLast[j] * _lastK![j][k];
                    dK[j][k] += dScoresLast[j] * _lastQ![lastPos][k];
                }
            }

            for (int a = 0; a < d; a++)
            {
                for (int b = 0; b < d; b++)
                {
                    weights.Wq[a, b] -= lr * _lastInput[lastPos][a] * dQLast[b];
                }
            }

            for (int j = 0; j <= lastPos; j++)
            {
                for (int a = 0; a < d; a++)
                {
                    for (int b = 0; b < d; b++)
                    {
                        weights.Wk[a, b] -= lr * _lastInput[j][a] * dK[j][b];
                        weights.Wv[a, b] -= lr * _lastInput[j][a] * dV[j][b];
                    }
                }
            }

            float[][] dX = new float[n][];
            for (int j = 0; j < n; j++)
            {
                dX[j] = new float[d];
            }

            for (int j = 0; j <= lastPos; j++)
            {
                for (int a = 0; a < d; a++)
                {
                    for (int b = 0; b < d; b++)
                    {
                        dX[j][a] += dK[j][b] * weights.Wk[a, b];
                        dX[j][a] += dV[j][b] * weights.Wv[a, b];
                    }
                }
            }

            for (int a = 0; a < d; a++)
            {
                for (int b = 0; b < d; b++)
                {
                    dX[lastPos][a] += dQLast[b] * weights.Wq[a, b];
                }
            }

            return dX;
        }

        private float[][] Multiply(float[][] input, float[,] matrix, int d)
        {
            int n = input.Length;
            float[][] result = new float[n][];
            for (int i = 0; i < n; i++)
            {
                result[i] = new float[d];
                for (int j = 0; j < d; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        result[i][j] += input[i][k] * matrix[k, j];
                    }
                }
            }
            return result;
        }

        private static float[][] DeepClone(float[][] src)
        {
            float[][] result = new float[src.Length][];
            for (int i = 0; i < src.Length; i++)
            {
                result[i] = (float[])src[i].Clone();
            }
            return result;
        }
    }
}
