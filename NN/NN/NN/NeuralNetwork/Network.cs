using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
namespace HundoNN
{

    [Serializable]
    public class Network
    {
        public int InputDimension{ get { return layers.First().Neurons.Length; } }
        public int OutputDimension { get { return layers.Last().Neurons.Length; } }

        private List<Layer> layers { get; set; } = new List<Layer>();//此網路的層數
        private List<List<List<double>>> TrainData { get; set; }
        private RegularizationHandler _regularizationMap;
        private RegularizationHandler _regularizationDerivative;
        private LossHandler _lossMap;
        private LossHandler _lossDerivative;




        /// <summary>
        /// 設定Regularization Funciton
        /// </summary>
        /// <param name="regularization"></param>
        public void SetRegularization(IRegularization regularization)
        {
            _regularizationMap = regularization.Map;
            _regularizationDerivative = regularization.Derivative;
        }


        /// <summary>
        /// 設定Loss Function
        /// </summary>
        /// <param name="loss"></param>
        public void SetLoss(ILoss loss)
        {
            _lossMap = loss.Map;
            _lossDerivative = loss.Derivative;
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

        /// <summary>
        /// 加入一層全連接層
        /// </summary>
        /// <param name="n"></param>
        /// <param name="activation"></param>
        /// <param name="normalization"></param>
        public void AddLayer(int n, IActivation activation, INormalization normalization)
        {
            AddLayer(new Layer(n, activation, normalization));
        }

        /// <summary>
        /// 加入一層全連接層
        /// </summary>
        /// <param name="n"></param>
        public void AddLayer(int n)
        {
            AddLayer(new Layer(n));
        }

        /// <summary>
        /// 加入一層全連接層
        /// </summary>
        /// <param name="n"></param>
        /// <param name="activation"></param>
        public void AddLayer(int n, IActivation activation)
        {
            AddLayer(new Layer(n, activation));
        }


        /// <summary>
        /// 載入神經層
        /// </summary>
        /// <param name="layer"></param>
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
            for (int i = 0; i < layers.Last().Neurons.Length; i++)
            {
                //TODO: 要能切換lossDer
                layers.Last().Neurons[i].OutputDer =
                    _lossDerivative(layers.Last().Neurons[i].Output, target[i]);
            }
            
            //從隱藏層最後一層往前回饋直到 輸入層(不含)
            for (int currentLayerIndex = layers.Count - 1; currentLayerIndex >= 1; currentLayerIndex--)
            {
                Layer currentLayer = layers[currentLayerIndex];

                //Todo: 這裡是實驗性的code
                double[] totalInputs = new double[currentLayer.Neurons.Length];

                for (int neuronIndex = 0; neuronIndex < currentLayer.Neurons.Length; neuronIndex++)
                {
                    Neuron currentNeuron = currentLayer.Neurons[neuronIndex];
                    totalInputs[neuronIndex] = currentNeuron.TotalInput;

                }

                //Todo:實驗性的計算softmax導函數
                double[] newTotalInputs = currentLayer.NormalizationDerivative(totalInputs);

                //算出每個神經元的誤差導數
                //1: 總輸入
                //2: 輸入權重
                for (int neuronIndex = 0; neuronIndex < currentLayer.Neurons.Length; neuronIndex++)
                {
                    Neuron currentNeuron = currentLayer.Neurons[neuronIndex];
                    
                    //Todo:這也是實驗的一環
                    currentNeuron.TotalInput = newTotalInputs[neuronIndex];

                    //神經元.輸入導數 = 神經元.輸出導數 * 激勵導數函式(神經元.總輸入)
                    currentNeuron.InputDer =
                        currentNeuron.OutputDer *
                        currentNeuron.ActivationDerivative(currentNeuron.TotalInput);
                    //神經元.累積輸入導數 += 神經元.輸入導數
                    currentNeuron.AccInputDer += currentNeuron.InputDer;
                    //神經元.累積輸入導數的數量
                    currentNeuron.NumAccumulatedDers++;
                }
                //進入神經元的每個權重的誤差導數
                for (int neuronIndex = 0; neuronIndex < currentLayer.Neurons.Length; neuronIndex++)
                {
                    Neuron currentNeuron = currentLayer.Neurons[neuronIndex];
                    for (int i = 0; i < currentNeuron.Weight.Length; i++)
                    {

                        Neuron prevNeuron = layers[currentLayerIndex - 1].Neurons[i];
                        //if (currentNeuron.isDead[i]) continue;
                        currentNeuron.WeightErrorDer[i] = currentNeuron.InputDer * prevNeuron.Output;
                        currentNeuron.WeightAccErrorDer[i] += currentNeuron.WeightErrorDer[i];
                        currentNeuron.WeightNumAccumulatedDers[i]++;

                    }
                }
                if (currentLayerIndex == 1) continue;
                //前一層
                Layer prevLayer = layers[currentLayerIndex - 1];
                //前一層的每一顆
                for (int i = 0; i < prevLayer.Neurons.Length; i++)
                {
                    Neuron prevNeuron = prevLayer.Neurons[i];
                    //計算每個神經元的誤差導數
                    prevNeuron.OutputDer = 0;
                    //前一層的每根輸出連結 == 這一層的輸入連結
                    for (int j = 0; j < currentLayer.Neurons.Length; j++)
                    {

                        prevNeuron.OutputDer +=
                            currentLayer.Neurons[j].Weight[i] *
                            currentLayer.Neurons[j].InputDer;
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
                for (int currentNeuronIndex = 0; currentNeuronIndex < currentLayer.Neurons.Length; currentNeuronIndex++)
                {
                    Neuron neuron = currentLayer.Neurons[currentNeuronIndex];
                    //更新這個神經元的bias
                    if (neuron.NumAccumulatedDers > 0)
                    {
                        neuron.Bias -= learningRate * neuron.AccInputDer / neuron.NumAccumulatedDers;
                        neuron.AccInputDer = 0;
                        neuron.NumAccumulatedDers = 0;
                    }
                    //更新每個進入這個神經元的權重
                    for (int i = 0; i < neuron.Weight.Length; i++)
                    {
                        double regulDer = _regularizationDerivative(neuron.Weight[i]);
                        if (neuron.WeightNumAccumulatedDers[i] > 0)
                        {
                            //用 E對w的微分調整權重
                            neuron.Weight[i] = neuron.Weight[i] -
                                (learningRate / neuron.WeightNumAccumulatedDers[i]) * neuron.WeightAccErrorDer[i];
                            //根據正則化(Regularization)進一步更新權重
                            //目前沒有使用
                            double newLinkWeight = neuron.Weight[i] -
                                (learningRate * regularizationRate) * regulDer;

                            neuron.Weight[i] = newLinkWeight;

                            //用完清掉
                            neuron.WeightAccErrorDer[i] = 0;
                            neuron.WeightNumAccumulatedDers[i] = 0;
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
        public double[] FeedForward(params double[] vs)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                vs = layers[i].Forward(vs);
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
        /// 取得平均誤差
        /// </summary>
        /// <param name="TargetData"></param>
        /// <returns></returns>
        private double GetLoss()
        {
            double loss = 0;
            for (int i = 0; i < TrainData.Count(); i++)
            {
                //正向計算結果
                double[] output = FeedForward(TrainData[i][0].ToArray());
                for (int j = 0; j < output.Length; j++)
                {
                    loss += _lossMap(output[j], TrainData[i][1][j]);
                }
            }
            return loss / TrainData.Count();
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
            Random random = new Random();
            for (uint e = 0; e < epoch; e++)
            {
                for (int i = 0; i < TrainData.Count; i++)
                {
                    TrainData.Sort((x,y)=>random.Next(-1,2));
                    //正向計算結果
                    this.FeedForward(TrainData[i][0].ToArray());
                    //倒傳遞
                    this.Backward(TrainData[i][1].ToArray());
                    if ((e + 1) % batchSize == 0)
                    {
                        //更新權重
                        this.UpdateWeights(learningRate, 0);
                    }
                }
                if ((e % showPerEpoch) == 0)
                {
                    Show(e);
                }
            }
        }


        /// <summary>
        /// 顯示
        /// </summary>
        /// <param name="ep"></param>
        private void Show(ulong ep)
        {
            Console.WriteLine($"Epoch {ep}: Loss {GetLoss().ToString("0.00000")}");
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
            network._regularizationMap = typeof(IRegularization).GetMethod(lines[0]).CreateDelegate(typeof(RegularizationHandler), Regularization.Function) as RegularizationHandler;
            network._regularizationDerivative = typeof(IRegularization).GetMethod(lines[1]).CreateDelegate(typeof(RegularizationHandler), Regularization.Function) as RegularizationHandler;
            network._lossMap = typeof(ILoss).GetMethod(lines[2]).CreateDelegate(typeof(LossHandler), Loss.Function) as LossHandler;
            network._lossDerivative = typeof(ILoss).GetMethod(lines[3]).CreateDelegate(typeof(LossHandler), Loss.Function) as LossHandler;
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
        public static bool LoadFile(string path, out string data)
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
            stringBuilder.AppendLine(_regularizationMap.Method.Name);
            stringBuilder.AppendLine(_regularizationDerivative.Method.Name);
            stringBuilder.AppendLine(_lossMap.Method.Name);
            stringBuilder.AppendLine(_lossDerivative.Method.Name);
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



        public Network(IRegularization regularization, ILoss loss)
        {
            _lossMap = loss.Map;
            _lossDerivative = loss.Derivative;
            _regularizationMap = regularization.Map;
            _regularizationDerivative = regularization.Derivative;
            layers = new List<Layer>();
            TrainData = new List<List<List<double>>>();
        }
        public Network() : this(Regularization.Function.None, Loss.Function.MeanSquareError)
        {
        }

        /// <summary>
        /// Loss預設使用MSE
        /// </summary>
        /// <param name="regularization"></param>
        public Network(IRegularization regularization) : this(regularization, Loss.Function.MeanSquareError)
        {
        }
        /// <summary>
        /// Regularization預設不使用
        /// </summary>
        /// <param name="loss"></param>
        public Network(ILoss loss) : this(Regularization.Function.None, loss)
        {
        }
    }
}