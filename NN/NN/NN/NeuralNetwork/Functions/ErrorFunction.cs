using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public partial class Functions : IErrorFunction
    {
        public double MeanSquareError(double output, double target)
        {
            return 0.5 * Math.Pow(output - target, 2);
        }

        public double MeanSquareErrorDerivative(double output, double target)
        {
            return output - target;
        }
    }
}
