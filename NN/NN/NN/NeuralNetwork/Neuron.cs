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

        internal double[] input { get; set; }
        //權重
        public double[] weight { get; set; }
        //下一層的 輸入
        //outputs,
        //權重的 誤差導數
        internal double[] weightErrorDer;
        //權重的 累積誤差導數
        internal double[] weightAccErrorDer;
        //權重的 累計誤差導數的數量
        internal double[] weightNumAccumulatedDers;
        public double bias { get; set; }
        internal double output;
        internal double error;
        //總輸入
        internal double totalInput;
        //這個神經元的總輸入的誤差導數
        internal double inputDer;
        //這個神經元的輸出的誤差導數
        internal double outputDer;
        //這個神經元的總輸入的累積誤差導數
        internal double accInputDer;
        //累積誤差的數量
        internal double numAccumulatedDers;

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
            input = new double[n];
            weight = new double[n];
            weightErrorDer = new double[n];
            weightAccErrorDer = new double[n];
            weightNumAccumulatedDers = new double[n];
            bias = new double();
            ActivationMap = Activation.Function.Linear.Map;
            ActivationDerivative = Activation.Function.Linear.Derivative;
            Randomize(n);
        }

        /// <summary>
        /// 輸入層
        /// </summary>
        public Neuron() : this(1)
        {
            bias = 0;
            weight[0] = 1;
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
                input[0] = value[index];
                return input[0];
            }
            totalInput = bias;
            for (int i = 0; i < value.Length; i++)
            {
                input[i] = value[i];
                totalInput += value[i] * weight[i];
            }
            output = ActivationMap(totalInput);
            return output;
        }

        /// <summary>
        /// 初始化數值
        /// </summary>
        /// <param name="n"></param>
        public void Randomize(int n)
        {
            rdm = new Random(GetHashCode());
            bias = rdm.NextDouble() ;
            for (int i = 0; i < n; i++)
            {
                weight[i] = rdm.NextDouble() - 0.5;
            }
        }

        public string Save()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Neuron");
            stringBuilder.AppendLine(ActivationMap.Method.Name);
            stringBuilder.AppendLine(ActivationDerivative.Method.Name);
            stringBuilder.AppendLine(IsInput.ToString());
            stringBuilder.AppendLine(bias.ToString());
            stringBuilder.AppendLine(weight.Length.ToString());
            foreach (var w in weight)
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
            neuron.bias = double.Parse(lines[3]);
            int len = int.Parse(lines[4]);
            neuron.weight = new double[len];
            for (int i = 0; i < len; i++)
                neuron.weight[i] = double.Parse(lines[5+i]);
            neuron.input = new double[len];
            neuron.weightErrorDer = new double[len];
            neuron.weightAccErrorDer = new double[len];
            neuron.weightNumAccumulatedDers = new double[len];

            return neuron;
        }
    }
}
