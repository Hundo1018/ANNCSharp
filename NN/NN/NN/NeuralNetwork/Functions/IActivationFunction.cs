using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public delegate double ActivationHandler(params double[] x);

    public interface IActivation
    {
        double Map(params double[] x);
        double Derivative(params double[] x);
    }
}
