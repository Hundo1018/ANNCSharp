using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
namespace HundoNN
{
    public class Neuron
    {
        //public delegate double ActivationHandler (double x);
        //public bool[] isDead { get; set; }

        //權重
        public double[] Weight { get; set; }
        //下一層的 輸入
        //outputs,
        //權重的 誤差導數
        internal double[] WeightErrorDer { get; set; }
        //權重的 累積誤差導數
        internal double[] WeightAccErrorDer { get; set; }
        //權重的 累計誤差導數的數量
        internal double[] WeightNumAccumulatedDers { get; set; }
        public double Bias { get; set; }
        internal double Output { get; set; }
        internal double Error { get; set; }
        //總輸入
        internal double TotalInput { get; set; }
        //這個神經元的總輸入的誤差導數
        internal double InputDer { get; set; }
        //這個神經元的輸出的誤差導數
        internal double OutputDer { get; set; }
        //這個神經元的總輸入的累積誤差導數
        internal double AccInputDer { get; set; }
        //累積誤差的數量
        internal double NumAccumulatedDers { get; set; }

        public bool IsInput { get; set; }
        private Random rdm;

        //激勵函數跟他的導函數
        public ActivationHandler ActivationMap { get; set; }
        public ActivationHandler ActivationDerivative { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n">前一級輸出數量</param>
        public Neuron(int n)
        {
            Weight = new double[n];
            WeightErrorDer = new double[n];
            WeightAccErrorDer = new double[n];
            WeightNumAccumulatedDers = new double[n];
            Bias = new double();
            ActivationMap = Activation.Function.Linear.Map;
            ActivationDerivative = Activation.Function.Linear.Derivative;
            Randomize(n);
        }

        /// <summary>
        /// 輸入層
        /// </summary>
        public Neuron() : this(1)
        {
            Bias = 0;
            Weight[0] = 1;
            IsInput = true;
        }

        /// <summary>
        /// 輸入
        /// </summary>
        /// <param name="value">前一層所有輸出</param>
        /// <returns></returns>
        public double Forward(int index, params double[] value)
        {
            if (IsInput)
            {
                return value[index];
            }

            TotalInput = Bias;
            for (int i = 0; i < value.Length; i++)
            {
                TotalInput += value[i] * Weight[i];
            }
            Output = ActivationMap(TotalInput);
            Normalization.Function.Softmax.Map(value);
            return Output;
        }

        /// <summary>
        /// 初始化數值
        /// </summary>
        /// <param name="n"></param>
        public void Randomize(int n)
        {
            rdm = new Random(GetHashCode());
            Bias = rdm.NextDouble() ;
            for (int i = 0; i < n; i++)
            {
                Weight[i] = (rdm.NextDouble() - 0.5)*2;
            }
        }

        public string Save()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Neuron");
            stringBuilder.AppendLine(ActivationMap.Method.Name);
            stringBuilder.AppendLine(ActivationDerivative.Method.Name);
            stringBuilder.AppendLine(IsInput.ToString());
            stringBuilder.AppendLine(Bias.ToString());
            stringBuilder.AppendLine(Weight.Length.ToString());
            foreach (var w in Weight)
            {
                stringBuilder.AppendLine(w.ToString());
            }
            return stringBuilder.ToString();
        }

        /// <summary>
        /// 從字串資料讀取並回傳一個物件
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static Neuron Parse(string data)
        {
            Neuron neuron = new Neuron();
            string[] lines = data.Split(new string[] { "\r\n" }, StringSplitOptions.None);
            neuron.ActivationMap = typeof(IActivation).GetMethod(lines[0]).CreateDelegate(typeof(ActivationHandler), Activation.Function) as ActivationHandler;
            neuron.ActivationDerivative = typeof(IActivation).GetMethod(lines[1]).CreateDelegate(typeof(ActivationHandler), Activation.Function) as ActivationHandler;

            //neuron.ActivationMap = typeof(IActivation).GetMethod(lines[0]).CreateDelegate(typeof(IActivation), Functions.ActivationHandler) as ActivationHandler;
            //neuron.ActivationDerivative = typeof(IActivation).GetMethod(lines[1]).CreateDelegate(typeof(IActivation), Functions.ActivationHandler) as ActivationHandler;


            neuron.IsInput = bool.Parse(lines[2]);
            neuron.Bias = double.Parse(lines[3]);
            int len = int.Parse(lines[4]);
            neuron.Weight = new double[len];
            for (int i = 0; i < len; i++)
                neuron.Weight[i] = double.Parse(lines[5+i]);
            neuron.WeightErrorDer = new double[len];
            neuron.WeightAccErrorDer = new double[len];
            neuron.WeightNumAccumulatedDers = new double[len];

            return neuron;
        }
    }
}
