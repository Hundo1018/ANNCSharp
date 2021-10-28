using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double Activation(double x);

    public interface IActivationFunction
    {
        double Sigmoid(double x);
        double SigmoidDerivative(double x);
        double ReLU(double x);
        double ReLUDerivative(double x);
        double LeakyReLU(double x);
        double LeakyReLUDerivative(double x);
        double Linear(double x);
        double LinearDerivative(double x);
        //double SoftMax(params double[] x);
    }
}
