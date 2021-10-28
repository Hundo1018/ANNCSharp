using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections;
using NeuralNetwork;
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
        if (Network.LoadFile(XOR_FILE_NAME, out data))
        {
            network = Network.Parse(data);
        }
        else
        {
            network = new Network();
            network.AddLayer(new Layer(2));
            network.AddLayer(new Layer(5, Functions.Activation.ReLU, Functions.Activation.ReLUDerivative));
            network.AddLayer(new Layer(1, Functions.Activation.ReLU, Functions.Activation.ReLUDerivative));
            network.SetErrorFunction(Functions.Error.MeanSquareError, Functions.Error.MeanSquareErrorDerivative);
        }

        #region CreateData
        Random random = new Random();
        int testData = 1000;
        int scale = 10000;
        List<double[]> XorInput = new List<double[]>();
        List<double[]> XorLable = new List<double[]>();
        for (int i = 0; i < testData - 4; i++)
        {
            double[] xorI = new double[2];
            double[] xorL = new double[1];

            int xi0 = random.Next(0, scale + 1);
            int xi1 = random.Next(0, scale + 1);
            double yi = (xi0 ^ xi1) / (double)scale;

            XorInput.Add(new double[2] { ((double)xi0) / (double)scale, ((double)xi1) / (double)scale });
            XorLable.Add(new double[1] { yi });
        }
        XorInput.Add(new double[2] { 0, 0 });
        XorInput.Add(new double[2] { 0, 1 });
        XorInput.Add(new double[2] { 1, 0 });
        XorInput.Add(new double[2] { 1, 1 });
        XorLable.Add(new double[1] { 0 });
        XorLable.Add(new double[1] { 1 });
        XorLable.Add(new double[1] { 1 });
        XorLable.Add(new double[1] { 0 });
        #endregion

        for (int i = 0; i < testData; i++)
        {
            network.AddTrainData(XorInput[i].ToArray(), XorLable[i].ToArray());
        }

        network.Fit(0.03, 20, 1000, 100);

        var result0 = network.Forward(new double[] { 0, 0 })[0];
        var result1 = network.Forward(new double[] { 0, 1 })[0];
        var result2 = network.Forward(new double[] { 1, 0 })[0];
        var result3 = network.Forward(new double[] { 1, 1 })[0];

        Console.WriteLine($"{result0},{result1},{result2},{result3}\n");

        network.SaveFile(XOR_FILE_NAME);
    }
}