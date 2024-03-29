﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HundoNN
{
    public delegate double ActivationHandler(double x);
    //public delegate double[] LayerActivationHandler(double[] x);

    public interface IActivation
    {
        double Map(double x);
        //double[] MapLayer(double[] x);
        double Derivative(double x);
    }
}
