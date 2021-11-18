using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    public abstract class FunctionSingleton<T> where T : FunctionSingleton<T>
    {
        private static readonly Lazy<T> _lazy = new Lazy<T>(() => Activator.CreateInstance(typeof(T), true) as T);
        public static T Function { get { return _lazy.Value; } }
    }
}
