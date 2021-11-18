using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double RegularizationHandler(double x);
    public interface IRegularization
    {
        double Map(double w);
        double Derivative(double w);
    }
}
