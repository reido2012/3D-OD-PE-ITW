### Documentation
[Github Repo](https://github.com/reido2012/3D-OD-PE-ITW)
[Paper](https://arxiv.org/pdf/1803.11493.pdf)

Use a Docker Image with the correct version of Python and Tensorflow, I don't recommend going through the effort of installing the Jupyter Notebook stuff. 

``` 
nvidia-docker run -v ~/selerio:/notebooks/selerio -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash`
``````

To __run as jupyter notebook__ just run the __command above without the /bin/bash__

A list of available TF images is located [here](https://hub.docker.com/r/tensorflow/tensorflow/tags/)


### Installation
Install requirements for this repo 
````
pip install .
cd pascal3d
pip install -r requirements.txt
pip install -e .
````
Clone Tensorflow Models repository
````
git clone https://github.com/tensorflow/models.git
````

Install TF Models + Slim
````
cd models/research/slim
pip install .

cd ../../official
pip install -r requirements.txt
````

### Dataset
Download the pascal3d dataset from [here](http://cvgl.stanford.edu/projects/pascal3d.html)

Creating a usable dataset happens in `/pascal3d/pascal3d/dataset.py`

Change DATASET_DIR at the top of dataset.py to the path to the PASCAL3D+ Dataset you just downloaded

__To create a tfrecods version of the dataset__
````
python create_tfrecords.py
````
This code reads data that comes in the __pascal3d format__, parses it and returns it in the form of TFRecords datasets

The data contained in the each record is as follows

__object_image__ -- The cropped and centered image of the object 
__output_vector__ -- The ground truth virtual control points (16) + dimensions(3)
__apply_blur__ -- Whether or not blur should be applied to this image
__object_class__ -- Class of the object
__data_id__ -- The id of the annotation this record is from
__object_index__ -- Tells us which object in the original image record is of 


### Training Models
To train a pose estimation model the code is in real_domain_cnn.py

```
python real_domain_cnn.py --model_dir="/where/you/want/model_to_be_saved/"
```

All the hyperparameters are global variables in real_domain_cnn.py. They're set to the parameters in the paper. 

(Should move the hyperparameters to a separate file as they are used again)

You will have to change the variable TFRECORDS_DIR to where the tfrecords have been saved (should be wherever you ran create_tfrecords.py)

You will also have to download the resnet v1_50 checkpoint from `http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` 
Change the variable RESNET_V1_50 to be the path to this file.

TensorBoard is used to monitor the training process.
```
tensorboard --logdir="Wherever/The/Model/Dir/is/"
```

### Evaluation
In __eval_metrics.py__ is where I pass the eval data to the model to see the progress. This will give you the MedError and Acc PI/6 accuracy (I think something is wrong with these metrics)

To see what possible command line arguments are
```
python eval_metrics.py --help
```

While the model is training I usually use the command below and try a tfrecords file from the training set (check for overfitting) and then the eval set.
```
python eval_metrics.py --generate_imgs=True --model_dir="Whater/Model/You_Want" --tfrecords_file='/where_you_saved/the/pascal_val.tfrecords'
```

If you do not set the generate imgs flag it will evaluate the entire dataset which takes a while. 

This will run images through the model, for each image I create an image that shows the 
- ground truth virtual control points, 
- predicted virtual control points
- virtual control points from a transformed unit cube


### Getting Models to Mobile 
The aim was to get my Tensorflow Model to run in Unity IOS

This is a bit tricky and you might have to try training the model you create with different versions of TensorFlow due to the compatability differences between TensorFlowSharp/ML-Agents Tensorflow Unity Plugin and actual Tensorflow. 

The first step is to convert your tensorflow to the Unity compatible .bytes format 

In the code you should change the base_path which is where you want your bytes graph to be saved and set the input_graph_path.

```
python tf_model_to_bytes.py
```

Follow this tutorial to set up unity and add the model to an Unity project [TensorFlow + Unity: How to set up a custom TensorFlow graph in Unity](https://blog.goodaudience.com/tensorflow-unity-how-to-set-up-a-custom-tensorflow-graph-in-unity-d65cc1bd1ab1)

You will have to change the script that she has. What I did was downloaded this github repo [here](https://github.com/Syn-McJ/TFClassify-Unity/) and modified it to use my models. I update the script so that it would draw the virtual control points rather than bounding boxes. (I'll put the code for this somewhere)

Use the optimized version of the model when you add it to the Unity project.

You should run into the same problem that I did.

The major problem will be that there are unknown ops

There are many different missing ops the error comes in this form.

`
2018-09-19 15:27:24.824020: E tensorflow/core/framework/op_kernel.cc:1242] OpKernel ('op: "DecodeProtoV2" device_type: "CPU"') for unknown op: DecodeProtoV2
`
some more example of unknown ops - PopulationCount, RightShift, BitwiseOr, MutableDenseHashTableV2, LookupTableImportV2, 

`
TFException: No OpKernel was registered to support Op 'ListDiff' with these attrs.  Registered devices: [CPU], Registered kernels:
  <no registered kernels>
`

This issue should have been addressed here but there still seems to be an issue

[TensorFlowSharp iOS - missing operations and rebuilding TF iOS plugins · Issue #268 · Unity-Technologies/ml-agents · GitHub](https://github.com/Unity-Technologies/ml-agents/issues/268)



### Useful Resources
[Creating Custom Estimators  |  TensorFlow](https://www.tensorflow.org/guide/custom_estimators)
[3D Reconstruction — OpenCV (PNP and Projection)](https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp)
[Importing Data (TFRecords Dataset)  |  TensorFlow](https://www.tensorflow.org/guide/datasets)
[ML-Agents Installation ](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

### Future Work
Clean up the code! lol
Figure out why the Acc PI/6 and MedError seem off
Learn color from RGB image and apply to 3D model

