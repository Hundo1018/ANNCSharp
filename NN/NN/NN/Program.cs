using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections;
using HundoNN;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text;

static class Program
{
    /// <summary>
    /// 應用程式的主要進入點。
    /// </summary>
    [STAThread]
    static void Main()
    {

        string XOR_FILE_NAME = "XOR.txt";
        string data = "";
        Network network;
        network = new Network();
        network.AddLayer(2);
        network.AddLayer(5, Activation.Function.ReLU);
        network.AddLayer(1, Activation.Function.ReLU);
        network.SetErrorFunction(Error.Function.MeanSquareError);
        network.SetRegularization(Regularization.Function.None);



        #region CreateData

        Random random = new Random();
        List<double[]> XorInput = new List<double[]>();
        List<double[]> XorLable = new List<double[]>();
        XorInput.Add(new double[2] { 0, 0 });
        XorInput.Add(new double[2] { 0, 1 });
        XorInput.Add(new double[2] { 1, 0 });
        XorInput.Add(new double[2] { 1, 1 });
        XorLable.Add(new double[1] { 0 });
        XorLable.Add(new double[1] { 1 });
        XorLable.Add(new double[1] { 1 });
        XorLable.Add(new double[1] { 0 });
        #endregion


        for (int i = 0; i < XorLable.Count(); i++)
        {
            network.AddTrainData(XorInput[i].ToArray(), XorLable[i].ToArray());
        }
        //Train
        network.Fit(0.03, 20, 100000, 1000);
        //Test

        var result0 = network.Forward(new double[] { 0, 0 })[0];
        var result1 = network.Forward(new double[] { 0, 1 })[0];
        var result2 = network.Forward(new double[] { 1, 0 })[0];
        var result3 = network.Forward(new double[] { 1, 1 })[0];
        Console.WriteLine($"{result0},{result1},{result2},{result3}\n");

        result0 = network.Forward(new double[] { 0.1, 0.1 })[0];
        result1 = network.Forward(new double[] { 0.1, 0.9 })[0];
        result2 = network.Forward(new double[] { 0.9, 0.1 })[0];
        result3 = network.Forward(new double[] { 0.9, 0.9 })[0];
        Console.WriteLine($"{result0},{result1},{result2},{result3}\n");

        result0 = network.Forward(new double[] { 0.2, 0.2 })[0];
        result1 = network.Forward(new double[] { 0.2, 0.8 })[0];
        result2 = network.Forward(new double[] { 0.8, 0.2 })[0];
        result3 = network.Forward(new double[] { 0.8, 0.8 })[0];
        Console.WriteLine($"{result0},{result1},{result2},{result3}\n");

        result0 = network.Forward(new double[] { 0.4, 0.4 })[0];
        result1 = network.Forward(new double[] { 0.4, 0.6 })[0];
        result2 = network.Forward(new double[] { 0.6, 0.4 })[0];
        result3 = network.Forward(new double[] { 0.6, 0.6 })[0];
        Console.WriteLine($"{result0},{result1},{result2},{result3}\n");
        //network.SaveFile(XOR_FILE_NAME);
    }
}
