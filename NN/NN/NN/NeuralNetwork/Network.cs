﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Text.Json;
namespace NeuralNetwork
{

    [Serializable]
    public class Network
    {

        private List<Layer> layers { get; set; } = new List<Layer>();//此網路的層數
        private List<List<List<double>>> TrainData { get; set; }
        private Regularization _regularizationFunction;
        private Regularization _regularizationFunctionDerivative;
        private Error _errorFunction;
        private Error _errorFunctionDerivative;

        public Network(Regularization regularizationFunction, Regularization regularizationFunctionDerivative)
        {
            _regularizationFunction = regularizationFunction;
            _regularizationFunctionDerivative = regularizationFunctionDerivative;
            layers = new List<Layer>();
            TrainData = new List<List<List<double>>>();
        }
        public Network() : this(Functions.Regularization.None, Functions.Regularization.NoneDerivative)
        {
        }

        /// <summary>
        /// 加入一層全連接層
        /// </summary>
        /// <param name="layer"></param>
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
            if (layers.Count() == 1)
            {
                return;
            }
            layers.Last().Connect(layers[layers.Count() - 2]);
        }

        internal void LoadLayer(Layer layer)
        {
            layers.Add(layer);
        }
        /// <summary>
        /// 倒傳遞但是不更新，而是先累積誤差
        /// </summary>
        /// <param name="target"></param>
        private void Backward(double[] target)
        {
            //輸出層需要使用誤差函數來定義誤差
            for (int i = 0; i < layers.Last().neurons.Length; i++)
            {
                layers.Last().neurons[i].outputDer =
                    _errorFunctionDerivative(layers.Last().neurons[i].output, target[i]);
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
                        currentNeuron.ActivateFunctionDerivative(currentNeuron.totalInput);
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
                        if (currentNeuron.IsDead(i))
                            continue;
                        
                        Neuron prevNeuron = layers[currentLayerIndex - 1].neurons[i];
                        //if (currentNeuron.isDead[i]) continue;
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

        /// <summary>
        /// 更新權重
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="regularizationRate"></param>
        private void UpdateWeights(double learningRate, double regularizationRate)
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
                        if (neuron.IsDead(i)) continue;
                        double regulDer = (_regularizationFunction == Functions.Regularization.None ? 0 : _regularizationFunctionDerivative(neuron.weight[i]));
                        if (neuron.weightNumAccumulatedDers[i] > 0)
                        {
                            //用 E對w的微分調整權重
                            neuron.weight[i] = neuron.weight[i] -
                                (learningRate / neuron.weightNumAccumulatedDers[i]) * neuron.weightAccErrorDer[i];
                            //根據正則化(Regularization)進一步更新權重
                            double newLinkWeight = neuron.weight[i] -
                                (learningRate * regularizationRate) * regulDer;
                            if (_regularizationFunction == Functions.Regularization.L1 &&
                                neuron.weight[i] * newLinkWeight < 0)
                            {
                                //根據正則化規則 連結權重小於0時將連結判為死亡
                                neuron.weight[i] = 0;
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

        /// <summary>
        /// 前向傳播
        /// </summary>
        /// <param name="vs"></param>
        /// <returns></returns>
        public double[] Forward(params double[] vs)
        {
            for (int i = 0; i < this.layers.Count; i++)
            {
                vs = this.layers[i].Forward(vs);
            }
            return vs;
        }

        /// <summary>
        /// 加入訓練資料
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void AddTrainData(double[] input, double[] output)
        {
            List<List<double>> aPairData = new List<List<double>>();
            aPairData.Add(input.ToList());
            aPairData.Add(output.ToList());
            this.TrainData.Add(aPairData);
        }

        /// <summary>
        /// 取得誤差
        /// </summary>
        /// <param name="TargetData"></param>
        /// <returns></returns>
        private double GetLoss()
        {
            double loss = 0;
            for (int i = 0; i < TrainData.Count(); i++)
            {
                //正向計算結果
                double output = this.Forward(TrainData[i][0].ToArray())[0];
                loss += _errorFunction(output, TrainData[i][1][0]);
            }
            return loss / TrainData.Count();
        }



        public void SetErrorFunction(Error errorFunction, Error errorFunctionDerivative)
        {
            _errorFunction = errorFunction;
            _errorFunctionDerivative = errorFunctionDerivative;
        }

        /// <summary>
        /// 訓練模型
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="batchSize"></param>
        /// <param name="epoch"></param>
        /// <param name="showPerEpoch"></param>
        public void Fit(double learningRate, uint batchSize, uint epoch, uint showPerEpoch)
        {
            for (uint e = 0; e < epoch; e++)
            {
                for (int i = 0; i < TrainData.Count; i++)
                {
                    //正向計算結果
                    this.Forward(TrainData[i][0].ToArray());
                    //倒傳遞
                    this.Backward(TrainData[i][1].ToArray());
                    if ((e + 1) % batchSize == 0)
                    {
                        //更新權重
                        this.UpdateWeights(learningRate, 1);
                    }
                }
                if ((e % showPerEpoch) == 0)
                {
                    Show(e);
                }
            }
        }
        private void Show(uint ep)
        {
            Console.WriteLine($"Epoch {ep}: Loss {GetLoss().ToString("0.00000")}\n");
        }


        /// <summary>
        /// 從字串轉換成Network
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static Network Parse(string data)
        {
            Network network = new Network();
            string[] networkLines = data.Split(new string[] { "Network\r\n" }, StringSplitOptions.None);
            string[] layerLines = networkLines[1].Split(new string[] { "Layer\r\n" }, StringSplitOptions.None);
            string[] lines = layerLines[0].Split(new string[] { "\r\n" }, StringSplitOptions.None);
            network._regularizationFunction = typeof(IRegularizationFunction).GetMethod(lines[0]).CreateDelegate(typeof(Regularization), Functions.Regularization) as Regularization;
            network._regularizationFunctionDerivative = typeof(IRegularizationFunction).GetMethod(lines[1]).CreateDelegate(typeof(Regularization), Functions.Regularization) as Regularization;
            network._errorFunction = typeof(IErrorFunction).GetMethod(lines[2]).CreateDelegate(typeof(Error), Functions.Error) as Error;
            network._errorFunctionDerivative = typeof(IErrorFunction).GetMethod(lines[3]).CreateDelegate(typeof(Error), Functions.Error) as Error;
            int len = int.Parse(lines[4]);
            //Layer layer = new Layer(lines[0]);
            for (int i = 1; i <= len; i++)
            {
                Layer layer = Layer.Parse(layerLines[i]);
                network.LoadLayer(layer);
            }
            return network;
        }
        
        /// <summary>
        /// 從位置讀取字串資料
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static bool LoadFile(string path,out string data)
        {
            data = "";
            try
            {
                using (FileStream fileStream = new FileStream(path, FileMode.Open, FileAccess.Read))
                {
                    using (StreamReader streamReader = new StreamReader(fileStream, Encoding.Default))
                    {
                        data = streamReader.ReadToEnd();
                        streamReader.Close();
                    }
                    fileStream.Close();
                }
            }
            catch (Exception)
            {
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// 儲存資料
        /// </summary>
        /// <returns></returns>
        public string SaveData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Network");
            stringBuilder.AppendLine(_regularizationFunction.Method.Name);
            stringBuilder.AppendLine(_regularizationFunctionDerivative.Method.Name);
            stringBuilder.AppendLine(_errorFunction.Method.Name);
            stringBuilder.AppendLine(_errorFunctionDerivative.Method.Name);
            stringBuilder.AppendLine(layers.Count().ToString());
            for (int i = 0; i < layers.Count(); i++)
            {
                stringBuilder.AppendLine(layers[i].Save());
            }
            return stringBuilder.ToString();
        }
        
        /// <summary>
        /// 存檔
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public void SaveFile(string path)
        {
            string data = SaveData();
            using (FileStream fileStream = new FileStream(path, FileMode.Create, FileAccess.Write))
            {
                using (StreamWriter streamWriter = new StreamWriter(fileStream, Encoding.Default))
                {
                    streamWriter.Write(data);
                    streamWriter.Close();
                }
                fileStream.Close();
            }
        }
    }
}