using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public partial class Functions : IRegularizationFunction
    {
        public double None(double w)
        {
            return w;
        }
        public double NoneDerivative(double w)
        {
            return (w < 0 ? -1 : (w > 0 ? 1 : 0));
        }

        public double L1(double w)
        {
            return Math.Abs(w);
        }
        public double L1Derivative(double w)
        {
            return (w < 0 ? -1 : (w > 0 ? 1 : 0));
        }

        public double L2(double w)
        {
            return 0.5 * w * w;
        }

        public double L2Derivative(double w)
        {
            return w;
        }
    }
}
