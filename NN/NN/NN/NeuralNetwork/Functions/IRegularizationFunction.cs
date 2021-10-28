using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double Regularization(double x);

    public interface IRegularizationFunction
    {
        double None(double w);
        double NoneDerivative(double w);
        double L1(double w);
        double L1Derivative(double w);
        double L2(double w);
        double L2Derivative(double w);
    }
}
