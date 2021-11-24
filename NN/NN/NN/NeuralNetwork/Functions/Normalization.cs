using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public class Normalization : FunctionSingleton<Normalization>
    {
        public Softmax Softmax { get; } = Softmax.Function;
        public NoneNormalization None { get; } = NoneNormalization.Function;
    }
    public class Softmax : FunctionSingleton<Softmax>, INormalization
    {
        public double[] Derivative(double[] x)
        {
            throw new NotImplementedException();
        }

        public double[] Map(double[] x)
        {
            double[] exp = x;
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Math.Exp(x[i]);
                sum += exp[i];
            }
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = exp[i] / sum;
            }
            return exp;
        }
    }
    public class NoneNormalization : FunctionSingleton<NoneNormalization>, INormalization
    {
        public double[] Derivative(double[] x)
        {
            return x;
        }
        public double[] Map(double[] x)
        {
            return x;
        }
    }
}