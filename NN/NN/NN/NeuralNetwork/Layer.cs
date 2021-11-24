using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
namespace HundoNN
{
    // TODO: 多一種最尾端的Layer來做整層激勵函數的計算 Ex: SoftMax + CrossEntropy

    public class Layer
    {
        public Neuron[] neurons { get; set; }//此層的神經元數量
        private ActivationHandler ActivationMap { get; set; }
        private ActivationHandler ActivationDerivative { get; set; }
        private NormalizationHandler NormalizationMap { get; set; }
        private NormalizationHandler NormalizationDerivative { get; set; }
        public Layer(int n, IActivation activateFunction, INormalization normalizationFunction)
        {
            neurons = new Neuron[n];
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
                neurons[i] = new Neuron();
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
            int n = this.neurons.Length;
            double[] output = new double[n];
            for (int i = 0; i < n; i++)
            {
                output[i] = neurons[i].Forward(i, value);
            }
            output = NormalizationMap(output);
            return output;
        }

        /// <summary>
        /// 主體向下連接一層
        /// </summary>
        /// <param name="nextLayer">目標連接層</param>
        public void Connect(Layer lastLayer)
        {
            for (int i = 0; i < this.neurons.Length; i++)
            {
                neurons[i] = new Neuron(lastLayer.neurons.Length);
                neurons[i].ActivationMap = ActivationMap;
                neurons[i].ActivationDerivative = ActivationDerivative;
            }
        }

        public string Save()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Layer");
            stringBuilder.AppendLine(neurons.Length.ToString());
            stringBuilder.AppendLine(NormalizationMap.Method.Name);
            stringBuilder.AppendLine(NormalizationDerivative.Method.Name);

            foreach (var neuron in neurons)
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
            layer.neurons = new Neuron[len];
            layer.NormalizationMap = typeof(INormalization).GetMethod(lines[1]).CreateDelegate(typeof(NormalizationHandler), Normalization.Function) as NormalizationHandler;
            layer.NormalizationDerivative = typeof(INormalization).GetMethod(lines[2]).CreateDelegate(typeof(NormalizationHandler), Normalization.Function) as NormalizationHandler;
            for (int i = 3; i <= len; i++)
            {
                Neuron neuron = Neuron.Parse(lines[i]);
                layer.neurons[i - 1] = neuron;
            }
            layer.ActivationMap = layer.neurons[0].ActivationMap;
            layer.ActivationDerivative = layer.neurons[0].ActivationDerivative;
            return layer;
        }
    }
}