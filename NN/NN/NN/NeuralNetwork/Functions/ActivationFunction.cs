using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public partial class Functions : IActivationFunction
    {

        public double ReLU(double x)
        {
            return Math.Max(0, x);
        }
        public double ReLUDerivative(double x)
        {
            return (x <= 0 ? 0 : 1);
        }
        public double LeakyReLU(double x)
        {
            return Math.Max(0.01 * x, x);
        }
        public double LeakyReLUDerivative(double x)
        {
            return (x > 0 ? 1 : 0);
        }
        public double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public double SigmoidDerivative(double x)
        {
            x = Sigmoid(x);
            return x * (1 - x);
        }
        public double Linear(double x)
        {
            return x;
        }
        public double LinearDerivative(double x)
        {
            return 1;
        }
    }
}
