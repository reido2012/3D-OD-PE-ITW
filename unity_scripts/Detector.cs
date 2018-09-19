using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TensorFlow;
using UnityEngine;

#if UNITY_ANDROID && !UNITY_EDITOR
TensorFlowSharp.Android.NativeBinding.Init();
#endif

namespace TFClassify
{
    public class BoxOutline
    {
        public float YMin { get; set; } = 0;
        public float XMin { get; set; } = 0;
        public float YMax { get; set; } = 0;
        public float XMax { get; set; } = 0;
        public string Label { get; set; }
        public float Score { get; set; }
    }

    public class EstimatedPose{
        public float[] vcPoints { get; set; }
        public float[] dims { get; set; }
    }

    public class Detector
    {
        private static int IMAGE_MEAN = 117;
        private static float IMAGE_STD = 1;

        private int inputSize;
        private TFGraph graph;
        private string[] labels;

        public Detector(byte[] model, string[] labels, int inputSize)
        {
            
            this.labels = labels;
            this.inputSize = inputSize;
            this.graph = new TFGraph();
            this.graph.Import(new TFBuffer(model));
        }


        public Task<float[]> DetectAsync(Color32[] data)
        {
            return Task.Run(() =>
            {
                using (var session = new TFSession(this.graph))
                using (var tensor = TransformInput(data, this.inputSize, this.inputSize))
                {
                    var runner = session.GetRunner();
                    runner.AddInput(this.graph["input"][0], tensor)
                          .Fetch(this.graph["2d_predictions"][0]);
                    
                    var output = runner.Run();

                    var points = (float[])output[0].GetValue(jagged: false);
                    Console.WriteLine("Points");
                    Console.WriteLine(points);

                    foreach(var ts in output)
                    {
                        ts.Dispose();
                    }

                    return points;
                }
            });
        }


        public static TFTensor TransformInput(Color32[] pic, int width, int height)
        {
            byte[] floatValues = new byte[width * height * 3];

            for (int i = 0; i < pic.Length; ++i)
            {
                var color = pic[i];

                floatValues [i * 3 + 0] = (byte)((color.r - IMAGE_MEAN) / IMAGE_STD);
                floatValues [i * 3 + 1] = (byte)((color.g - IMAGE_MEAN) / IMAGE_STD);
                floatValues [i * 3 + 2] = (byte)((color.b - IMAGE_MEAN) / IMAGE_STD);
            }

            TFShape shape = new TFShape(1, width, height, 3);

            return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
        }

    }
}