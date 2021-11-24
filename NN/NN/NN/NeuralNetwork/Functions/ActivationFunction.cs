using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
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
        public double Map(double x)
        {
            return Math.Max(0, x);
        }
        public double Derivative(double x)
        {
            return (x <= 0 ? 0 : 1);
        }
    }
    public class LeakyReLU : FunctionSingleton<LeakyReLU>, IActivation
    {
        public double Map(double x)
        {
            return Math.Max(0.01 * x, x);
        }
        public double Derivative(double x)
        {
            return (x > 0 ? 1 : 0);
        }
    }
    public class Sigmoid : FunctionSingleton<Sigmoid>, IActivation
    {
        public double Map(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public double Derivative(double x)
        {
            x = Map(x);
            return x * (1 - x);
        }
    }
    public class Linear : FunctionSingleton<Linear>, IActivation
    {
        public double Map(double x)
        {
            return x;
        }
        public double Derivative(double x)
        {
            return 1;
        }
    }
}