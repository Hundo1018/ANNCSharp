using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public class Error : FunctionSingleton<Error>
    {
        public MeanSquareError MeanSquareError { get; } = MeanSquareError.Function;
    }
    public class MeanSquareError : FunctionSingleton<MeanSquareError>, IError
    {
        public double Derivative(double output, double target)
        {
            return output - target;
        }

        public double Map(double output, double target)
        {
            return 0.5 * Math.Pow(output - target, 2);
        }
    }

}
