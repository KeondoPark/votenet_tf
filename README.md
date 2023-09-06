# Introduction
This is the code archive for the paper ["PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators"](https://dl.acm.org/doi/abs/10.1145/3583120.3587045) (IPSN 2023)
The model is based on VoteNet architecture. (https://github.com/facebookresearch/votenet).

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
To train the model, you can use the following command.
```
python train_tf.py --config_path config/file/path
```

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

# Pre-trained models
Please download from the following link for tensorflow checkpoints. You need to create a folder named `tf_ckpt` and extract the downloaded files inside `tf_ckpt` folder. `model_id`  in config files should be consistent with the checkpoint folder name to restore the model.
| Model \ Dataset | SUN RGB-D | Scannet V2 |
|:---------------:|:---------:|:----------:|
|     baseline    | [link](https://mysnu-my.sharepoint.com/:u:/g/personal/gundo0102_seoul_ac_kr/EVZHxRQo2YVKuDKSdu288-gBExp8zsaUijckQCMs8nIFtw?e=dTez3D)      | [link](https://mysnu-my.sharepoint.com/:u:/g/personal/gundo0102_seoul_ac_kr/ERZ_zouUxHtEiCPTvCxXFE0Bxe6P35MLvPm1VvMQV0WB0A?e=3hcpph)       |
|    _PointSplit_ | [link](https://mysnu-my.sharepoint.com/:u:/g/personal/gundo0102_seoul_ac_kr/ETiJi4nn9-BLltzvz3ybfcwBU8jgdtfTFkCc1-eEMWP34Q?e=UYCA0W)      | [link](https://mysnu-my.sharepoint.com/:u:/g/personal/gundo0102_seoul_ac_kr/ERZ_zouUxHtEiCPTvCxXFE0Bxe6P35MLvPm1VvMQV0WB0A?e=3hcpph)       |


# Converting to tflite model (Quantization)
Related files are located under `tflite` folder. To convert any trained model into tflite models in *post-training quantization* way, use following command:
```
python create_tflite_p.py --config_path config/file/path
```

`q_gran` option in config file could be used to select the quantization granularity option for the last layers of voting and proposal module. 'semantic' option corresponds to role-based groupwise quantization. 'layer','group' or 'channel' option could be used for different quantization granularity. The converted tflite files are saved under `tflite/tflite_models` folder.

# Measure inference latency
Inference latency is measured on NVIDIA Jetson Nano (4GB) equipped with Google Coral EdgeTPU. To reproduce the results in our paper, one needs to prepare the devices and settings. 
- To install drivers necessary to use EdgeTPU on Jetson Nano, one can refer to this url: https://coral.ai/docs/m2/get-started/#4-run-a-model-on-the-edge-tpu 

- To install Tensorflow on Jetson Nano, follow the steps described in this url: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

- To install PyTorch on Jetson Nano, follow the steps described in this url: https://qengineering.eu/install-pytorch-on-jetson-nano.html

Once the environment setting is completed as explained above, one can run the inference with following command:
```
python demo_tf.py --config_path config/file/path
```

# Other 3D object detection baseline models
Please see other branches (groupfree2, repsurf) for implementation with other 3D object detection baseline models.
