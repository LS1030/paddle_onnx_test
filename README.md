# PaddleSeg 模型推理测试工程

## 问题描述

发现`paddleseg`的模型和使用`paddle2onnx`转换出来的`onnx`模型的推理结果不一致，精度差距较大。

不知道是因为什么原因导致的。

## 测试环境

### 飞桨版本

paddle2onnx              1.1.0
paddlepaddle-gpu         2.5.1.post120
paddleseg                2.9.0        

### 工程目录

```
.
├── images
│   └── test
├── labels
│   └── test
├── log
│   ├── infer.py.log
│   ├── onnx_calculate_miou.log
│   └── val.py.log
├── merge_images
├── model
│   ├── best_model
│   ├── deploy
│   │   ├── deploy.yaml
│   │   ├── model.onnx
│   │   ├── model.pdiparams
│   │   ├── model.pdiparams.info
│   │   └── model.pdmodel
│   └── seaformer_test_dataset_640x480.yml
├── onnx_inference
├── paddle_inference
├── README.md
├── script
│   ├── infer.py
│   ├── merge_images_filelist.py
│   ├── onnx_calculate_miou.py
│   ├── onnxruntime_infer_multi.py
│   └── val.py
├── test.txt
├── train.txt
└── val.txt
```

已按照飞桨的要求构建数据集，训练集train.txt和验证集val.txt为空，测试集test.txt包含11张图片。

模型目标为分割地面，只有两类，标签0为背景，标签1为地面。

模型存放在model目录:
- seaformer_test_dataset_640x480.yml  训练模型配置文件
- best_model                          训练好的paddle动态图模型
- deploy                              导出的模型文件夹
- deploy/deploy.yaml                  部署模型配置文件
- deploy/model.onnx                   onnx模型    
- deploy/model.pdiparams              paddle静态态图模型参数
- deploy/model.pdiparams.info         paddle静态态图模型参数信息
- deploy/model.pdmodel                paddle静态态图模型结构

推理脚本和miou计算脚本在script目录:
- val.py                        用于计算paddle模型推理的miou
- infer.py                      用于paddle模型推理，保存推理图片
- onnx_calculate_miou.py        用于计算onnx模型推理的miou
- onnxruntime_infer_multi.py    用于onnx模型推理，保存推理图片
- merge_images_filelist.py      用于合并原图、paddle推理图片、onnx推理图片

log文件夹为各脚本输出的log文件。

paddle_inference和onnx_inference为保存推理图片的目录。
merge_images为合并原图、paddle推理图片、onnx推理图片的目录。

## 复现步骤

### 1. 模型转换

导出静态图模型, 此命令在`PaddleSeg`目录下执行
```
python tools/export.py \
       --config dataset/ZM/test_dataset/model/seaformer_test_dataset_640x480.yml \
       --model_path dataset/ZM/test_dataset/model/best_model/model.pdparams \
       --save_dir dataset/ZM/test_dataset/model/deploy \
       --input_shape 1 3 480 640 \
       --output_op none
```

转换为onnx，此命令在`当前目录`下执行
```
paddle2onnx \
    --model_dir model/deploy \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --save_file model/deploy/model.onnx \
    --opset_version 12 \
    --enable_onnx_checker True
```

### 2. 计算miou

#### paddle

在`当前目录`下执行
```
python script/val.py --config model/seaformer_test_dataset_640x480.yml --model_path model/best_model/model.pdparams
```

输出结果为：
```
2024-03-28 17:09:54 [INFO]      [EVAL] #Images: 11 mIoU: 0.9510 Acc: 0.9878 Kappa: 0.9491 Dice: 0.9745
2024-03-28 17:09:54 [INFO]      [EVAL] Class IoU: 
[0.916 0.986]
2024-03-28 17:09:54 [INFO]      [EVAL] Class Precision: 
[0.9349 0.9966]
2024-03-28 17:09:54 [INFO]      [EVAL] Class Recall: 
[0.9783 0.9893]
```
#### onnx

在`当前目录`下执行
```
python script/onnx_calculate_miou.py \
    --config model/deploy/deploy.yaml \
    --model_path model/deploy/model.onnx \
    --image_root_path ./ \
    --image_filelist test.txt 
```

输出结果为：
```
mIoU: 89.13%
IoU per class:
Class 0: 80.66%
Class 1: 97.60%
```

### 3. 推理

#### paddle

在`当前目录`下执行
```
python script/infer.py \
    --config model/deploy/deploy.yaml \
    --image_root_path ./ \
    --image_filelist test.txt \
    --save_dir paddle_inference \
    --with_argmax
```

#### onnx

在`当前目录`下执行
```
python script/onnxruntime_infer_multi.py \
    --config model/deploy/deploy.yaml \
    --model_path model/deploy/model.onnx \
    --image_root_path ./ \
    --image_filelist test.txt \
    --save_dir onnx_inference
```

### 4. 合并图片

在`当前目录`下执行
```
python script/merge_images_filelist.py \
    --image_root_path ./ \
    --image_filelist test.txt \
    --paddle_inference_dir paddle_inference \
    --onnx_inference_dir onnx_inference \
    --save_dir merge_images
```

## 测试结果

通过执行上诉复现步骤，可以得到如下结果：
1. paddle miou为95.10%，onnx miou为89.13%，精度差距较大。
2. 推理图片对比结果与上述miou预期一致, 精度差距较大。

![推理结果对比](https://github.com/LS1030/paddle_onnx_test/blob/master/merge_images/back_20240112-111331_1009.png?raw=true "推理结果对比")

左边为原图，中间为paddleseg模型的推理结果，右边为onnx模型的推理结果。