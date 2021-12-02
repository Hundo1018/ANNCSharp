using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
namespace HundoNN
{
    public class Layer
    {
        public Neuron[] Neurons { get; set; }//此層的神經元數量
        private ActivationHandler ActivationMap { get; set; }
        //LayerActivationHandler layerActivationHandler;
        private ActivationHandler ActivationDerivative { get; set; }
        public NormalizationHandler NormalizationMap { get; set; }
        public NormalizationHandler NormalizationDerivative { get; set; }
        public Layer(int n, IActivation activateFunction, INormalization normalizationFunction)
        {
            Neurons = new Neuron[n];
            ActivationMap = activateFunction.Map;
            ActivationDerivative = activateFunction.Derivative;
            NormalizationMap = normalizationFunction.Map;
            NormalizationDerivative = normalizationFunction.Derivative;
        }
        public Layer(int n, IActivation activateFunction):this(n, activateFunction,Normalization.Function.None)
        {
        }


        /// <summary>
        /// 輸入層
        /// </summary>
        /// <param name="n"></param>
        public Layer(int n) : this(n, Activation.Function.Linear)
        {
            for (int i = 0; i < n; i++)
            {
                Neurons[i] = new Neuron();
            }
        }

        /// <summary>
        /// 給Load使用
        /// </summary>
        private Layer()
        {

        }

        /// <summary>
        /// 前饋
        /// </summary>
        /// <param name="value"></param>
        public double[] Forward(params double[] value)
        {
            int n = this.Neurons.Length;
            double[] output = new double[n];

            for (int i = 0; i < n; i++)
            {
                output[i] = Neurons[i].Forward(i, value);
            }
            output = NormalizationMap(output);
            for (int i = 0; i < n; i++)
            {
                Neurons[i].Output = output[i];
            }
            return output;
        }

        /// <summary>
        /// 主體向下連接一層
        /// </summary>
        /// <param name="nextLayer">目標連接層</param>
        public void Connect(Layer lastLayer)
        {
            for (int i = 0; i < this.Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(lastLayer.Neurons.Length);
                Neurons[i].ActivationMap = ActivationMap;
                Neurons[i].ActivationDerivative = ActivationDerivative;
            }
        }

        public string Save()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Layer");
            stringBuilder.AppendLine(Neurons.Length.ToString());
            stringBuilder.AppendLine(NormalizationMap.Method.Name);
            stringBuilder.AppendLine(NormalizationDerivative.Method.Name);

            foreach (var neuron in Neurons)
            {
                stringBuilder.AppendLine(neuron.Save());
            }
            return stringBuilder.ToString();
        }

        public static Layer Parse(string data)
        {
            Layer layer = new Layer();
            string[] lines = data.Split(new string[] { "Neuron\r\n" }, StringSplitOptions.None);
            int len = int.Parse(lines[0]);
            layer.Neurons = new Neuron[len];
            layer.NormalizationMap = typeof(INormalization).GetMethod(lines[1]).CreateDelegate(typeof(NormalizationHandler), Normalization.Function) as NormalizationHandler;
            layer.NormalizationDerivative = typeof(INormalization).GetMethod(lines[2]).CreateDelegate(typeof(NormalizationHandler), Normalization.Function) as NormalizationHandler;
            for (int i = 3; i <= len; i++)
            {
                Neuron neuron = Neuron.Parse(lines[i]);
                layer.Neurons[i - 1] = neuron;
            }
            layer.ActivationMap = layer.Neurons[0].ActivationMap;
            layer.ActivationDerivative = layer.Neurons[0].ActivationDerivative;
            return layer;
        }
    }
}