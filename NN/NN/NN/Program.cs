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
        network = new Network(Regularization.Function.None, Loss.Function.MeanSquareError);
        network.AddLayer(2);
        network.AddLayer(8, Activation.Function.ReLU, Normalization.Function.None);

        network.AddLayer(4, Activation.Function.ReLU, Normalization.Function.None);


        #region CreateData

        Random random = new Random();
        List<double[]> XorInput = new List<double[]>();
        List<double[]> XorLable = new List<double[]>();
        XorInput.Add(new double[] { 0, 0 });
        XorInput.Add(new double[] { 0, 1 });
        XorInput.Add(new double[] { 1, 0 });
        XorInput.Add(new double[] { 1, 1 });
        XorLable.Add(new double[] { 0, 0, 0, 1 });
        XorLable.Add(new double[] { 0, 0, 1, 0 });
        XorLable.Add(new double[] { 0, 1, 0, 0 });
        XorLable.Add(new double[] { 1, 0, 0, 0 });
        #endregion


        for (int i = 0; i < XorLable.Count(); i++)
        {
            network.AddTrainData(XorInput[i].ToArray(), XorLable[i].ToArray());
        }
        //Train
        network.Fit(0.03, 1, 300000, 500);
        //Test
        var result0 = network.FeedForward(new double[] { 0, 0 });
        var result1 = network.FeedForward(new double[] { 0, 1 });
        var result2 = network.FeedForward(new double[] { 1, 0 });
        var result3 = network.FeedForward(new double[] { 1, 1 });
        Console.WriteLine($"{result0[0]},{result0[1]},{result0[2]},{result0[3]}\n");
        Console.WriteLine($"{result1[0]},{result1[1]},{result1[2]},{result1[3]}\n");
        Console.WriteLine($"{result2[0]},{result2[1]},{result2[2]},{result2[3]}\n");
        Console.WriteLine($"{result3[0]},{result3[1]},{result3[2]},{result3[3]}\n");



    }
}