# Real-time 3D Multi-person Pose Estimation Demo on jetson TX2

This repository is oringnally forked from [Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch](https://github.com/daniil-osokin/lightweight-human-pose-estimation-3d-demo.pytorch), and modified to use TensorRT engine on jetson TX2 device, about 110ms for one frame.

<p align="center">
  <img src="data/human_pose_estimation_3d_demo.jpg" />
</p>

> The major part of this work was done by [Mariia Ageeva](https://github.com/marrmar), when she was the :top::rocket::fire: intern at Intel.

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Inference with TensorRT](#inference-tensorrt)

## Requirements
* Python 3.5 (or above)
* CMake 3.10 (or above)
* C++ Compiler (g++ or MSVC)
* OpenCV 4.0 (or above)
* TensorRT engine

> [tensorRT](https://developer.nvidia.com/tensorrt) for fast inference on nvidia GPU(for me it is **TX2**).

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```
2. Build `pose_extractor` module:
```
python setup.py build_ext
```
3. Add build folder to `PYTHONPATH`:
```
export PYTHONPATH=pose_extractor/build/:$PYTHONPATH
```

## Pre-trained model <a name="pre-trained-model"/>

Pre-trained model is available at [Google Drive](https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view?usp=sharing).

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python demo.py --model human-pose-estimation-3d.pth --video 0
```
> Camera can capture scene under different view angles, so for correct scene visualization, please pass camera extrinsics and focal length with `--extrinsics` and `--fx` options correspondingly (extrinsics sample format can be found in data folder). In case no camera parameters provided, demo will use the default ones.

## Inference with TensorRT 
   
   
To run with TensorRT, it is necessary to convert checkpoint to onnx format and then change to tensorrt engine. I converted it in `models/` for **nvidia Jetson TX2** and named it as `human-pose-estimation-3d.trt`.

```py
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 30 
    builder.max_batch_size = 1 
    builder.fp16_mode = True 
    with open('human-pose-estimation-3d.onnx', 'rb') as model: 
        parser.parse(model.read()) 
    engine = builder.build_cuda_engine(network) 
    with open('human-pose-estimation-3d.trt', "wb") as f: 
        f.write(engine.serialize())
```

To run the demo with TensorRT inference, pass `--use-tensorrt` option and specify device to infer on:
```
python demo.py --model models/human-pose-estimation-3d.trt --device GPU --use-tensorrt --video 0
```
