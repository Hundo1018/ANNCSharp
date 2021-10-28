using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public partial class Functions: IActivationFunction, IErrorFunction, IRegularizationFunction
    {
        private readonly static Functions _functions = new Functions();
        public static IActivationFunction Activation
        {
            get { return _functions; } 
        }
        public static IErrorFunction Error
        {
            get { return _functions; }
        }
        public static IRegularizationFunction Regularization
        {
            get { return _functions; }
        }
        Functions()
        {
        }
    }
}


