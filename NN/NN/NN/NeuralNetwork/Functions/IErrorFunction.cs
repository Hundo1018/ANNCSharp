using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double ErrorHandler(double output, double target);

    public interface IError
    {
        double Map(double output, double target);
        double Derivative(double output, double target);
    }
}
