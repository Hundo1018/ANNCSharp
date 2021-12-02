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
        double expSum = 0;
        private double S(double x)
        {
            return Math.Exp(x) / expSum;
        }
        public double[] Derivative(double[] x)
        {
            double[] exp = x;
            expSum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                expSum += Math.Exp(x[i]);
            }
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < x.Length; j++)
                {
                    if (i==j)
                    {
                        exp[i] = S(x[i]) * (1 - S(x[i]));
                    }
                    else 
                    {
                        exp[i] = -S(x[i]) * S(x[j]);
                    }
                }
            }
            return exp;
        }

        public double[] Map(double[] x)
        {
            double[] exp = x;
            expSum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                expSum += Math.Exp( exp[i]);
            }
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = S(exp[i]);
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