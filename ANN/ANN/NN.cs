using System;
using System.Collections.Generic;
using System.Linq;

namespace Test//ANN
{
    using NN= NeuroNetwork;
    using ANN = NeuroNetwork;
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //建立網路
            NN.Network net = new NN.Network();
            net.regularizationFunctionType = NN.RegularizationFunctionType.None;
            net.errorFunctionType = NN.ErrorFunctionType.MSE;
            //建立輸入層(2個神經元)
            net.AddLayer(new NN.Layer.InputLayer(2));
            //建立隱藏層(2個神經元)
            net.AddLayer(new NN.Layer.FullyConnectLayer(2));
            //建立輸出層(1個神經元)
            net.AddLayer(new NN.Layer.FullyConnectLayer(1));
            //設定網路使用的激勵函數
            net.layers[0].activateFunction = NN.ActivateFunctionType.ReLU;
            net.layers[1].activateFunction = NN.ActivateFunctionType.ReLU;
            net.layers[2].activateFunction = NN.ActivateFunctionType.Linear;
            #region Xor訓練資料
            for (int i = 0; i <= 1; i++)
            {
                for (int j = 0; j <= 1; j++)
                {
                    net.AddTrainData(new double[] { i, j }, new double[] { i ^ j });
                }
            }
            #endregion



            //一萬回
            for (int ep = 0; ep < 10000; ep++)
            {

                net.OneStep(0.03, 0);

            }

        }
    }

}
namespace NeuroNetwork
{

    public class Neuron
    {
        public bool[] isDead;

        public double[] input, weight,
            //下一層的 輸入
            //outputs,
            //權重的 誤差導數
            weightErrorDer,
            //權重的 累積誤差導數
            weightAccErrorDer,
            //權重的 累計誤差導數的數量
            weightNumAccumulatedDers;
        public double bias, output, error,
            //總輸入
            totalInput,
            //這個神經元的總輸入的誤差導數
            inputDer,
            //這個神經元的輸出的誤差導數
            outputDer,
            //這個神經元的總輸入的累積誤差導數
            accInputDer,
            //累積誤差的數量
            numAccumulatedDers;
        Random rdm;
        internal IActivateFunction activateFunction;
        public Neuron(int n)
        {
            input = new double[n];
            weight = new double[n];
            isDead = new bool[n];
            weightErrorDer = new double[n];
            weightAccErrorDer = new double[n];
            weightNumAccumulatedDers = new double[n];
            bias = new double();
            Randomize(n);
        }
        public double Input(params double[] value)
        {
            totalInput = bias;
            for (int i = 0; i < value.Length; i++)
            {
                input[i] = value[i];
                totalInput += value[i] * weight[i];
            }
            output = activateFunction.Output(totalInput);
            return output;
        }

        public void Randomize(int n)
        {
            rdm = new Random();
            if (n == 1)
            {
                bias = 0;
                weight[0] = 1;
                return;
            }
            bias = 0.1;
            for (int i = 0; i < n; i++)
            {
                weight[i] = rdm.NextDouble() - 0.5;
            }
        }

    }
    public abstract class Layer
    {
        public Neuron[] neurons;//此層的神經元數量
        public ActivateFunctionType activateFunction
        {
            set
            {
                switch (value)
                {
                    case ActivateFunctionType.Linear:
                        neurons.Select(x => x.activateFunction = new ActivateFunction.Linear()); ;
                        break;
                    case ActivateFunctionType.Sigmoid:
                        neurons.Select(x => x.activateFunction = new ActivateFunction.Sigmoid()); ;
                        break;
                    case ActivateFunctionType.ReLU:
                        neurons.Select(x => x.activateFunction = new ActivateFunction.ReLU()); ;
                        break;
                    case ActivateFunctionType.LeakyReLU:
                        neurons.Select(x => x.activateFunction = new ActivateFunction.LeakyReLU()); ;
                        break;
                    default:
                        break;
                }
            }
        }
        public Layer(int n, ActivateFunctionType activateFunction)
        {
            neurons = new Neuron[n];
            this.activateFunction = activateFunction;
        }
        public Layer(int n)
        {
            neurons = new Neuron[n];
        }
        /// <summary>
        /// 前饋
        /// </summary>
        /// <param name="value"></param>
        public abstract double[] Forward(params double[] value);

        /// <summary>
        /// 主體向下連接一層
        /// </summary>
        /// <param name="nextLayer">目標連接層</param>
        public abstract void Connect(Layer lastLayer);
        public class InputLayer : Layer
        {
            public InputLayer(int n) : base(n)
            {
                neurons = new Neuron[n];
                for (int i = 0; i < n; i++)
                {
                    neurons[i] = new Neuron(1);
                }
            }

            public override void Connect(Layer lastLayer)
            {
            }
            /// <summary>
            /// 這裡的輸入是網路外部，即第一層輸入
            /// </summary>
            /// <param name="value"></param>
            /// <returns></returns>
            public override double[] Forward(params double[] inputValue)
            {
                int n = this.neurons.Length;
                double[] result = new double[n];
                for (int i = 0; i < n; i++)
                {
                    result[i] = neurons[i].Input(inputValue[i]);
                }

                return result;
            }
        }
        public class FullyConnectLayer : Layer
        {
            /// <summary>
            /// FullyConnect - 全連接層
            /// </summary>
            /// <param name = "n" > 此層有n個神經元 </ param >
            public FullyConnectLayer(int n) : base(n)
            {
                neurons = new Neuron[n];
            }

            public override void Connect(Layer lastLayer)
            {
                for (int i = 0; i < this.neurons.Length; i++)
                {
                    neurons[i] = new Neuron(lastLayer.neurons.Length);
                }
            }

            public override double[] Forward(params double[] lastLayerOutputs)
            {
                int n = this.neurons.Length;
                double[] output = new double[n];
                for (int i = 0; i < n; i++)
                {
                    output[i] = (neurons[i].Input(lastLayerOutputs));
                }
                return output;
            }
        }
        public class ReLuLayer
        {

        }
        public class SoftMax
        {

        }
    }
    public class Network
    {
        public List<Layer> layers;//此網路的層數
        public List<List<List<double>>> TrainData;
        public List<List<List<double>>> TestData;

        public RegularizationFunctionType regularizationFunctionType
        {
            set
            {
                switch (value)
                {
                    case RegularizationFunctionType.None:
                        regularizationFunction = new RegularizationFunction.None();
                        break;
                    case RegularizationFunctionType.L1:
                        regularizationFunction = new RegularizationFunction.L1();
                        break;
                    case RegularizationFunctionType.L2:
                        regularizationFunction = new RegularizationFunction.L2();
                        break;
                    default:
                        break;
                }
            }
        }
        public ErrorFunctionType errorFunctionType
        {
            set
            {
                switch (value)
                {
                    case ErrorFunctionType.MSE:
                        errorFunction = new ErrorFunction.MSE();
                        break;
                    default:
                        break;
                }
            }
        }
        private IRegularizationFunction regularizationFunction;
        private IErrorFunction errorFunction;
        public Network()
        {
            layers = new List<Layer>();
            TrainData = new List<List<List<double>>>();
            regularizationFunctionType = RegularizationFunctionType.None;
            errorFunctionType = ErrorFunctionType.MSE;
        }
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);

            for (int i = 0; i < layers.Count - 1; i++)
            {
                layers[i + 1].Connect(layers[i]);
            }
        }

        public void Backward(double[] target)
        {
            //輸出層需要使用誤差函數來定義誤差
            for (int i = 0; i < layers.Last().neurons.Length; i++)
            {
                layers.Last().neurons[i].outputDer =
                    errorFunction.Derivative(layers.Last().neurons[i].output, target[i]);
            }
            //從隱藏層最後一層往前回饋直到 輸入層(不含)
            for (int currentLayerIndex = layers.Count - 1; currentLayerIndex >= 1; currentLayerIndex--)
            {
                Layer currentLayer = layers[currentLayerIndex];
                //算出每個神經元的誤差導數
                //1: 總輸入
                //2: 輸入權重
                for (int neuronIndex = 0; neuronIndex < currentLayer.neurons.Length; neuronIndex++)
                {
                    Neuron currentNeuron = currentLayer.neurons[neuronIndex];

                    //神經元.輸入導數 = 神經元.輸出導數 * 激勵導數函式(神經元.總輸入)
                    currentNeuron.inputDer =
                        currentNeuron.outputDer *
                        currentNeuron.activateFunction.Derivative(currentNeuron.totalInput);
                    //神經元.累積輸入導數 += 神經元.輸入導數
                    currentNeuron.accInputDer += currentNeuron.inputDer;
                    //神經元.累積輸入導數的數量
                    currentNeuron.numAccumulatedDers++;
                }
                //進入神經元的每個權重的誤差導數
                for (int neuronIndex = 0; neuronIndex < currentLayer.neurons.Length; neuronIndex++)
                {
                    Neuron currentNeuron = currentLayer.neurons[neuronIndex];
                    for (int i = 0; i < currentNeuron.weight.Length; i++)
                    {
                        Neuron prevNeuron = layers[currentLayerIndex - 1].neurons[i];
                        if (currentNeuron.isDead[i]) continue;
                        currentNeuron.weightErrorDer[i] = currentNeuron.inputDer * prevNeuron.output;
                        currentNeuron.weightAccErrorDer[i] += currentNeuron.weightErrorDer[i];
                        currentNeuron.weightNumAccumulatedDers[i]++;

                    }
                }
                if (currentLayerIndex == 1) continue;
                //前一層
                Layer prevLayer = layers[currentLayerIndex - 1];
                //前一層的每一顆
                for (int i = 0; i < prevLayer.neurons.Length; i++)
                {
                    Neuron prevNeuron = prevLayer.neurons[i];
                    //計算每個神經元的誤差導數
                    prevNeuron.outputDer = 0;
                    //前一層的每根輸出連結 == 這一層的輸入連結
                    for (int j = 0; j < currentLayer.neurons.Length; j++)
                    {

                        prevNeuron.outputDer +=
                            currentLayer.neurons[j].weight[i] *
                            currentLayer.neurons[j].inputDer;
                    }

                }
            }

        }

        public void UpdateWeights(double learningRate, double regularizationRate)
        {
            for (int layerIndex = 1; layerIndex < layers.Count; layerIndex++)
            {
                Layer currentLayer = layers[layerIndex];
                for (int currentNeuronIndex = 0; currentNeuronIndex < currentLayer.neurons.Length; currentNeuronIndex++)
                {
                    Neuron neuron = currentLayer.neurons[currentNeuronIndex];
                    //更新這個神經元的bias
                    if (neuron.numAccumulatedDers > 0)
                    {
                        neuron.bias -= learningRate * neuron.accInputDer / neuron.numAccumulatedDers;
                        neuron.accInputDer = 0;
                        neuron.numAccumulatedDers = 0;
                    }
                    //更新每個進入這個神經元的權重
                    for (int i = 0; i < neuron.weight.Length; i++)
                    {
                        //Link link = neuron.inputLinks[i];
                        if (neuron.isDead[i]) continue;
                        double regulDer = (this.regularizationFunction.ToString() == new RegularizationFunction.None().ToString() ?
                            0 : this.regularizationFunction.Derivative(neuron.weight[i]));

                        if (neuron.weightNumAccumulatedDers[i] > 0)
                        {
                            //用 E對w的微分調整權重
                            neuron.weight[i] = neuron.weight[i] -
                                (learningRate / neuron.weightNumAccumulatedDers[i]) * neuron.weightAccErrorDer[i];
                            //根據正則化(Regularization)進一步更新權重
                            double newLinkWeight = neuron.weight[i] -
                                (learningRate * regularizationRate) * regulDer;
                            if (this.regularizationFunction == new RegularizationFunction.L1() &&
                                neuron.weight[i] * newLinkWeight < 0)
                            {
                                //根據正則化規則 連結權重小於0時將連結判為死亡
                                neuron.weight[i] = 0;
                                neuron.isDead[i] = true;
                            }
                            else
                            {
                                neuron.weight[i] = newLinkWeight;
                            }
                            neuron.weightAccErrorDer[i] = 0;
                            neuron.weightNumAccumulatedDers[i] = 0;
                        }
                    }
                }
            }
        }
        public double[] Forward(params double[] vs)
        {
            for (int i = 0; i < this.layers.Count; i++)
            {
                double[] temp = this.layers[i].Forward(vs);
                vs = temp;
            }
            return vs;
        }

        public void AddTrainData(double[] input, double[] output)
        {
            List<List<double>> aPairData = new List<List<double>>();
            aPairData.Add(input.ToList());
            aPairData.Add(output.ToList());
            this.TrainData.Add(aPairData);
        }

        public double GetLoss(List<List<List<double>>> TargetData)
        {
            double loss = 0;
            for (int i = 0; i < TargetData.Count; i++)
            {
                //正向計算結果
                double output = this.Forward(TrainData[i][0].ToArray())[0];
                loss += errorFunction.Error(output, TrainData[i][1][0]);
            }
            return (loss / TargetData.Count);
        }

        public void OneStep(double learningRate, double regularizationRate)
        {
            for (int i = 0; i < TrainData.Count; i++)
            {
                //正向計算結果
                List<double[]> outputs = new List<double[]>();
                outputs.Add(this.Forward(TrainData[i][0].ToArray()));
                //倒傳遞
                this.Backward(TrainData[i][1].ToArray());
                //更新權重
                this.UpdateWeights(learningRate, regularizationRate);
            }
        }

    }


    public enum ActivateFunctionType { Linear, Sigmoid, ReLU, LeakyReLU };
    internal interface IActivateFunction
    {
        double Output(double x);
        double Derivative(double x);

    }
    internal class ActivateFunction
    {
        public class Sigmoid : IActivateFunction
        {
            #region Singleton
            private static Sigmoid instance = null;
            public static Sigmoid Instance
            {
                get
                {
                    return instance ?? (instance = new Sigmoid());
                }
            }
            internal Sigmoid() { }
            #endregion


            public static double Output(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
            public static double Derivative(double x)
            {
                x = Sigmoid.Output(x);
                return x * (1 - x);
            }
            double IActivateFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IActivateFunction.Output(double x)
            {
                return Output(x);
            }
        }
        public class ReLU : IActivateFunction
        {
            #region Singleton
            private static ReLU instance = null;
            public static ReLU Instance
            {
                get
                {
                    return instance ?? (instance = new ReLU());
                }
            }
            internal ReLU() { }
            #endregion

            public static double Output(double x)
            {
                return Math.Max(0, x);
            }
            public static double Derivative(double x)
            {
                return (x <= 0 ? 0 : 1);
            }
            double IActivateFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IActivateFunction.Output(double x)
            {
                return Output(x);
            }
        }
        public class LeakyReLU : IActivateFunction
        {
            #region Singleton
            private static LeakyReLU instance = null;
            public static LeakyReLU Instance
            {
                get
                {
                    return instance ?? (instance = new LeakyReLU());
                }
            }
            internal LeakyReLU() { }
            #endregion
            public static double Output(double x)
            {
                return Math.Max(0.01 * x, x);
            }
            public static double Derivative(double x)
            {
                return (x > 0 ? 1 : 0);
            }
            double IActivateFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IActivateFunction.Output(double x)
            {
                return Output(x);
            }
        }
        public class Linear : IActivateFunction
        {
            #region Singleton
            private static Linear instance = null;
            public static Linear Instance
            {
                get
                {
                    return instance ?? (instance = new Linear());
                }
            }
            internal Linear() { }
            #endregion
            public static double Output(double x)
            {
                return x;
            }
            public static double Derivative(double x)
            {
                return 1;
            }
            double IActivateFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IActivateFunction.Output(double x)
            {
                return Output(x);
            }
        }
    }


    public enum ErrorFunctionType { MSE };
    internal interface IErrorFunction
    {
        double Error(double output, double target);
        double Derivative(double output, double target);
    }
    internal class ErrorFunction
    {

        public class MSE : IErrorFunction
        {
            #region Singleton
            private static MSE instance = null;
            public static MSE Instance
            {
                get
                {
                    return instance ?? (instance = new MSE());
                }
            }
            internal MSE() { }
            #endregion
            public static double Error(double output, double target)
            {
                return 0.5 * Math.Pow(output - target, 2);
            }
            public static double Derivative(double output, double target)
            {
                return output - target;
            }
            double IErrorFunction.Error(double output, double target)
            {
                return Error(output, target);
            }
            double IErrorFunction.Derivative(double output, double target)
            {
                return Derivative(output, target);
            }
        }
    }


    public enum RegularizationFunctionType { None, L1, L2 };
    internal interface IRegularizationFunction
    {
        double Output(double w);
        double Derivative(double w);
    }
    internal class RegularizationFunction
    {
        public class None : IRegularizationFunction
        {
            #region Singleton
            private static None instance = null;
            public static None Instance
            {
                get
                {
                    return instance ?? (instance = new None());
                }
            }
            internal None() { }
            #endregion
            public static double Output(double w)
            {
                return w;
            }
            public static double Derivative(double w)
            {
                return 0;
            }
            double IRegularizationFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IRegularizationFunction.Output(double x)
            {
                return Output(x);
            }
        }
        public class L1 : IRegularizationFunction
        {
            #region Singleton
            private static L1 instance = null;
            public static L1 Instance
            {
                get
                {
                    return instance ?? (instance = new L1());
                }
            }
            internal L1() { }
            #endregion
            public static double Output(double w)
            {
                return Math.Abs(w);
            }
            public static double Derivative(double w)
            {
                return (w < 0 ? -1 : (w > 0 ? 1 : 0));
            }
            double IRegularizationFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IRegularizationFunction.Output(double x)
            {
                return Output(x);
            }
        }
        public class L2 : IRegularizationFunction
        {
            #region Singleton
            private static L2 instance = null;
            public static L2 Instance
            {
                get
                {
                    return instance ?? (instance = new L2());
                }
            }
            internal L2() { }
            #endregion
            public static double Output(double w)
            {
                return 0.5 * w * w;
            }
            public static double Derivative(double w)
            {
                return w;
            }
            double IRegularizationFunction.Derivative(double x)
            {
                return Derivative(x);
            }
            double IRegularizationFunction.Output(double x)
            {
                return Output(x);
            }
        }
    }

}
