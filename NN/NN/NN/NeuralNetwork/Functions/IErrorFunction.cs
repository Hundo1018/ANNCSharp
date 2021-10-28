using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double Error(double output, double target);

    public interface IErrorFunction
    {
        double MeanSquareError(double output, double target);
        double MeanSquareErrorDerivative(double output, double target);

    }
}
