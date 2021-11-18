using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Activation : FunctionSingleton<Activation>
    {
        public ReLU ReLU { get; } = ReLU.Function;
        public LeakyReLU LeakyReLU { get; } = LeakyReLU.Function;
        public Linear Linear { get; } = Linear.Function;
        public Sigmoid Sigmoid { get; } = Sigmoid.Function;
    }


    
   public class ReLU : FunctionSingleton<ReLU>, IActivation
    {
        public double Map(params double[] x)
        {
            return Math.Max(0, x.First());
        }
        public double Derivative(params double[] x)
        {
            return (x.First() <= 0 ? 0 : 1);
        }
    }
    public class LeakyReLU : FunctionSingleton<LeakyReLU>, IActivation
    {
        public double Map(params double[] x)
        {
            return Math.Max(0.01 * x.First(), x.First());
        }
        public double Derivative(params double[] x)
        {
            return (x.First() > 0 ? 1 : 0);
        }
    }
    public class Sigmoid : FunctionSingleton<Sigmoid>, IActivation
    {
        public double Map(params double[] x)
        {
            return 1.0 / (1.0 + Math.Exp(-x.First()));
        }
        public double Derivative(params double[] x)
        {
            x[0] = Map(x[0]);
            return x[0] * (1 - x[0]);
        }
    }
    public class Linear : FunctionSingleton<Linear>, IActivation
    {
        public double Map(params double[] x)
        {
            return x.First();
        }
        public double Derivative(params double[] x)
        {
            return 1;
        }
    }
    public class Softmax : FunctionSingleton<Softmax>, IActivation
    {
        public double Derivative(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double Map(params double[] x)
        {
            throw new NotImplementedException();
        }
    }
}