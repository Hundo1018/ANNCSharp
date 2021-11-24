using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public delegate double LossHandler(double[] output, double[] target);
    public interface ILoss
    {
        double Map(double[] output, double[] target);
        double Derivative(double[] output, double[] target);
    }
}
