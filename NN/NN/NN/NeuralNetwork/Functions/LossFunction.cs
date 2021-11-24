using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    class LossFunction : FunctionSingleton<LossFunction>
    {
        public CrossEntropy CrossEntropy { get; } = CrossEntropy.Function;
        public NoneLoss NoneLoss { get; } = NoneLoss.Function;
    }
    public class CrossEntropy : FunctionSingleton<CrossEntropy>, ILoss
    {
        public double Derivative(double[] output, double[] target)
        {
            throw new NotImplementedException();
        }

        public double Map(double[] softMax, double[] target)
        {
            double result = 0;
            for (int i = 0; i < target.Length; i++)
            {
                result -= Math.Log(target[i] * softMax[i]);
            }
            return result;
        }
    }
    public class NoneLoss : FunctionSingleton<NoneLoss>, ILoss
    {
        public double Derivative(double[] output, double[] target)
        {
            throw new NotImplementedException();
        }

        public double Map(double[] output, double[] target)
        {
            throw new NotImplementedException();
        }
    }

}
