using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public delegate double[] NormalizationHandler(double[] x);

    public interface INormalization
    {
        double[] Map(double[] x);
        double[] Derivative(double[] x);
    }
}
