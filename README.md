# Introduction
This is the code archive for the paper ["PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators"](https://dl.acm.org/doi/abs/10.1145/3583120.3587045) (IPSN 2023)
This branch includes the implementation of PointSplit based on RepSurf (https://github.com/hancyran/RepSurf) and GroupFree3D model (https://github.com/zeliu98/Group-Free-3D).

![image](figs/Fig3.png)

# Installation
To run this code, you need both PyTorch and Tensorflow. The model is mainly built in Tensorflow 2.5.0, but some PyTroch utils (v.1.8) are also utilized. You also need access to GPUs and MATLAB is required to prepare SUN RGB-D dataset.

To install required package, create an anaconda environment using requirement file. A new anaconda environment named "pointsplit" is created.
```
conda env create -f requirements.yaml
```

One needs to compile the Tensorflow custom operations that are used in backbone network. 
- Go to `pointent2/tf_ops/sampling`
- run `bash tf_sampling_compile.sh` (Change `CUDA_ROOT` as appropriate in the script)
- Repeat for `pointent2/tf_ops/interpolation`, `pointent2/tf_ops/grouping`

# Training
Before training the model, need to prepare datasets(SUN RGB-D or Scannet V2). Please refer to `sunrgbd` or `scannet` folders for preparation. This also includes *PointPainting* which appends 2D semantic segmentation results on the input point cloud. 
RepSurf features are calculated per each point and stored as input data.


To train the model, refer to `rsgf_train.sh`.

## Code description
**Biased FPS** is implemented in `pointent2/tf_ops/sampling/tf_sampling_g.cu` - `farthestpointsamplingBgKernel` function. This method gives more weights on painted points to sample more painted points, i.e. sampling *biased* upon painted points.

**2-way set abstraction layer** aims to sample 2 different point sets from the input point cloud so that the model could be parallelized during inference. Please see `models/backbone_module_tf.py` for codes. `Pointnet2Backbone_p` class includes the training codes for 2-way SA layer and `Pointnet2Backbone_tflite` incldues the inference codes using tflite(as well as EdgeTPU).

Pretrained **DeeplabV3+** for *pointpainting* is located under `deeplab/saved_model`. This includes graph files(.pb) as well as tflite files(.tflite). They are used to prepare *pointpainted* point cloud for trainig or during inference.

## Description of configuration
Some sample config files are included in `configs` folder. The prefix *config_* imples the config files for training and *inf_* prefix implies the config files for inference.
```
- model-id: Unique model id
- dataset: The dataset used for training or inference, sunrgbd or scannet, default: sunrgbd
- use_painted: Whether or not to use pointpainting
- two_way: Whether or not to use 2-way set abstraction layer
- use_fp_mlp: Whether or not to use PointNet(SharedMLP) in FP Layers (Original votenet used FP MLP, but PointSplit does not)
- q_gran: Quantization granularity. semantic/channel/group/layer
- activation: Which activation to use, relu or relu6
- use_tflite: Whether or not to use tflite for inference (Inference only)
- tfltie_folder: If use tflite, in which folder tflite files are located (Inference only)
- use_edgetpu: If use tflite, run it on edgetpu or cpu (Inference only)
- use_multiThr: If use tflite and 2-way set abstraction, whether or not to use pipelining for inference (Inference only)
```

# Converting to tflite model (Quantization) / Measure inference latency
GroupFree / RepSurf baseline models are not intended to be deployed to Jetson Nano, and no tflite version are created.
Inference latency on jetson nano is not measured either.
