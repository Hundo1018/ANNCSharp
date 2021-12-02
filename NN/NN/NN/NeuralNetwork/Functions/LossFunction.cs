using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public class Loss : FunctionSingleton<Loss>
    {
        public MeanSquareError MeanSquareError { get; } = MeanSquareError.Function;
        public CrossEntropy CrossEntropy { get; } = CrossEntropy.Function;
    }
    public class MeanSquareError : FunctionSingleton<MeanSquareError>, ILoss
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
    public class CrossEntropy : FunctionSingleton<CrossEntropy>, ILoss
    {
        public double Derivative(double output, double target)
        {
            double result = -(target) / (output);
            return result;
        }

        public double Map(double output, double target)
        {
            double result = -(target) * Math.Log(output + double.Epsilon);
            return result;
        }
    }
}
