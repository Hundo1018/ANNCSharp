using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public class Regularization : FunctionSingleton<Regularization>
    {
        public NoneRegularization None { get; } = NoneRegularization.Function;
        public L1 L1 { get; } = L1.Function;
        public L2 L2 { get; } = L2.Function;
    }
    public class NoneRegularization : FunctionSingleton<NoneRegularization>, IRegularization
    {
        public double Derivative(double w)
        {
            return (w < 0 ? -1 : (w > 0 ? 1 : 0));
        }

        public double Map(double w)
        {
            return w;
        }
    }
    public class L1 : FunctionSingleton<L1>, IRegularization
    {
        public double Derivative(double w)
        {
            return (w < 0 ? -1 : (w > 0 ? 1 : 0));
        }

        public double Map(double w)
        {
            return Math.Abs(w);
        }
    }
    public class L2 : FunctionSingleton<L2>, IRegularization
    {
        public double Derivative(double w)
        {
            return w;
        }

        public double Map(double w)
        {
            return 0.5 * w * w;
        }
    }
}
