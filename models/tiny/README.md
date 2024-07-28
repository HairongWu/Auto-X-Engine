# Tiny Models

## ShuffleNetV2

The ShuffleNet series network is the lightweight network structure proposed by MEGVII. So far, there are two typical structures in this series network, namely, ShuffleNetV1 and ShuffleNetV2. A Channel Shuffle operation in ShuffleNet can exchange information between groups and perform end-to-end training. In the paper of ShuffleNetV2, the author proposes four criteria for designing lightweight networks, and designs the ShuffleNetV2 network according to the four criteria and the shortcomings of ShuffleNetV1.

### Accuracy, FLOPs and Parameters

| Models                               | Top1    | Top5    | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ShuffleNetV2                         | 0.688   | 0.885   | 0.694             |                   | 0.280        | 2.260             |
| ShuffleNetV2_x0_25                   | 0.499   | 0.738   |                   |                   | 0.030        | 0.600             |
| ShuffleNetV2_x0_33                   | 0.537   | 0.771   |                   |                   | 0.040        | 0.640             |
| ShuffleNetV2_x0_5                    | 0.603   | 0.823   | 0.603             |                   | 0.080        | 1.360             |
| ShuffleNetV2_x1_5                    | 0.716   | 0.902   | 0.726             |                   | 0.580        | 3.470             |
| ShuffleNetV2_x2_0                    | 0.732   | 0.912   | 0.749             |                   | 1.120        | 7.320             |
| ShuffleNetV2_swish                   | 0.700   | 0.892   |                   |                   | 0.290        | 2.260             |

### Inference speed and storage size based on SD855

| Models                               | Batch Size=1(ms) | Storage Size(M) |
|:--:|:--:|:--:|
| ShuffleNetV2                         | 10.941           | 9.000           |
| ShuffleNetV2_x0_25                   | 2.329            | 2.700           |
| ShuffleNetV2_x0_33                   | 2.643            | 2.800           |
| ShuffleNetV2_x0_5                    | 4.261            | 5.600           |
| ShuffleNetV2_x1_5                    | 19.352           | 14.000          |
| ShuffleNetV2_x2_0                    | 34.770           | 28.000          |
| ShuffleNetV2_swish                   | 16.023           | 9.100           |

### Inference speed based on T4 GPU

| Models            | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-----------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| ShuffleNetV2                | 1.95064               | 2.15928               | 2.97169               | 1.89436               | 2.26339               | 3.17615               |
| ShuffleNetV2_x0_25          | 1.43242               | 2.38172               | 2.96768               | 1.48698               | 2.29085               | 2.90284               |
| ShuffleNetV2_x0_33          | 1.69008               | 2.65706               | 2.97373               | 1.75526               | 2.85557               | 3.09688               |
| ShuffleNetV2_x0_5           | 1.48073               | 2.28174               | 2.85436               | 1.59055               | 2.18708               | 3.09141               |
| ShuffleNetV2_x1_5           | 1.51054               | 2.4565                | 3.41738               | 1.45389               | 2.5203                | 3.99872               |
| ShuffleNetV2_x2_0           | 1.95616               | 2.44751               | 4.19173               | 2.15654               | 3.18247               | 5.46893               |
| ShuffleNetV2_swish          | 2.50213               | 2.92881               | 3.474                 | 2.5129                | 2.97422               | 3.69357               |

### ImageNet1k

The quality and quantity of data often determine the performance of a model. In the field of image classification, data includes images and labels. In most cases, labeled data is scarce, so the amount of data is difficult to reach the level of saturation of the model. In order to enable the model to learn more image features, a lot of image transformation or data augmentation is required before the image enters the model, so as to ensure the diversity of input image data and ensure that the model has better generalization capabilities.

[ImageNet](https://image-net.org/) is a large visual database for visual target recognition research with over 14 million manually labeled images. ImageNet-1k is a subset of the ImageNet dataset, which contains 1000 categories with 1281167 images for the training set and 50000 for the validation set. Since 2010, ImageNet began to hold an annual image classification competition, namely, the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with ImageNet-1k as its specified dataset. To date, ImageNet-1k has become one of the most significant contributors to the development of computer vision, based on which numerous initial models of downstream computer vision tasks are trained.

| Dataset                                                      | Size of Training Set | Size of Test Set | Number of Category | Note |
| ------------------------------------------------------------ | -------------------- | ---------------- | ------------------ | ---- |
| [ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/) | 1.2M                 | 50k              | 1000               |      |

After downloading the data from official sources, organize it in the following format to train with the ImageNet1k dataset in PaddleClas.

```
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```

### Model Training

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
pip install paddleclas
```

If you want to use PaddlePaddle on CPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
pip install paddleclas
```

**Note:**
* If you have already installed CPU version of PaddlePaddle and want to use GPU version now, you should uninstall CPU version of PaddlePaddle and then install GPU version to avoid package confusion.
* You can also compile PaddlePaddle from source code, please refer to [PaddlePaddle Installation tutorial](http://www.paddlepaddle.org.cn/install/quick) to more compilation options.

After preparing the configuration file, The training process can be started in the following way.

```shell
python3 train.py \
    -c ./configs/ShuffelNet/ShuffleNetV2_x0_25.yaml \
    -o Arch.pretrained=False \
    -o Global.device=gpu
```

Among them, `-c` is used to specify the path of the configuration file, `-o` is used to specify the parameters needed to be modified or added, `-o Arch.pretrained=False` means to not using pre-trained models. `-o Global.device=gpu` means to use GPU for training. If you want to use the CPU for training, you need to set `Global.device` to `cpu`.

The output log examples are as follows:

- If mixup or cutmix is used in training, top-1 and top-k (default by 5) will not be printed in the log:

  ```
  ...
  [Train][Epoch 3/20][Avg]CELoss: 6.46287, loss: 6.46287
  ...
  [Eval][Epoch 3][Avg]CELoss: 5.94309, loss: 5.94309, top1: 0.01961, top5: 0.07941
  ...
  ```

- If mixup or cutmix is not used during training, in addition to the above information, top-1 and top-k (The default is 5) will also be printed in the log:

  ```
  ...
  [Train][Epoch 3/20][Avg]CELoss: 6.12570, loss: 6.12570, top1: 0.01765, top5: 0.06961
  ...
  [Eval][Epoch 3][Avg]CELoss: 5.40727, loss: 5.40727, top1: 0.07549, top5: 0.20980
  ...
  ```

During training, you can view loss changes in real time through `VisualDL`, see [VisualDL](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/en/extension/VisualDL_en.md) for details.

### Model Finetuning

After correcting config file, you can load pretrained model  weight to finetune. The command is as follows:

```shell
python3 train.py \
    -c ./configs/ShuffelNet/ShuffleNetV2_x0_25.yaml \
    -o Arch.pretrained=True \
    -o Global.device=gpu
```

Among them,`Arch.pretrained` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file. You can also set it into `True` to use pretrained weights that trained in ImageNet1k.


### Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```shell
python3 train.py \
    -c ./configs/ShuffelNet/ShuffleNetV2_x0_25.yaml \
    -o Global.checkpoints="./output/ShuffleNetV2_x0_25/epoch_5" \
    -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter.

**Note**:

- The `-o Global.checkpoints` parameter does not need to include the suffix of the checkpoints. The above training command will generate the checkpoints as shown below during the training process. If you want to continue training from the epoch `5`, Just set the `Global.checkpoints` to `../output/ShuffleNetV2_x0_25/epoch_5`, PaddleClas will automatically fill in the `pdopt` and `pdparams` suffixes. Files in the output directory are structured as follows：

  ```
  output
  ├── ShuffleNetV2_x0_25
  │   ├── best_model.pdopt
  │   ├── best_model.pdparams
  │   ├── best_model.pdstates
  │   ├── epoch_1.pdopt
  │   ├── epoch_1.pdparams
  │   ├── epoch_1.pdstates
      .
      .
      .
  ```

### Model Evaluation

The model evaluation process can be started as follows.

```shell
python3 eval.py \
    -c ./configs/ShuffleNetV2_x0_25.yaml \
    -o Global.pretrained_model=./output/ShuffleNetV2_x0_25/best_model
```

The above command will use `./configs/ShuffleNetV2_x0_25.yaml` as the configuration file to evaluate the model `./output/ShuffleNetV2_x0_25/best_model`. You can also set the evaluation by changing the parameters in the configuration file, or you can update the configuration with the `-o` parameter, as shown above.

Some of the configurable evaluation parameters are described as follows:

- `Arch.name`：Model name
- `Global.pretrained_model`：The path of the model file to be evaluated

## Training and Evaluation on Linux+ Multi-GPU

If you want to run PaddleClas on Linux with GPU, it is highly recommended to use `paddle.distributed.launch` to start the model training script(`train.py`) and evaluation script(`eval.py`), which can start on multi-GPU environment more conveniently.

### Model Training

The training process can be started in the following way. `paddle.distributed.launch` specifies the GPU running card number by setting `gpus`:

```shell
# PaddleClas initiates multi-card multi-process training via launch

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    train.py \
        -c ./configs/ShuffleNetV2_x0_25.yaml
```

### Model Finetuning

After configuring the yaml file, you can finetune it by loading the pretrained weights. The command is as below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    train.py \
        -c ./configs/ShuffleNetV2_x0_25.yaml \
        -o Arch.pretrained=True
```

Among them, `Arch.pretrained` is set to `True` or `False`. It also can be used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

### Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    train.py \
        -c ./configs/ShuffleNetV2_x0_25.yaml \
        -o Global.checkpoints="./output/ShuffleNetV2_x0_25/epoch_5" \
        -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. 

### Model Evaluation

The model evaluation process can be started as follows.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    eval.py \
        -c ./configs/ShuffleNetV2_x0_25.yaml \
        -o Global.pretrained_model=./output/ShuffleNetV2_x0_25/best_model
```


### Model Convert
Python3.7 is required.

```shell
pip install paddlelite
python export_model.py -c ./configs/ShuffleNetV2_x0_25.yaml -o Global.pretrained_model=./pretrain/ShuffleNetV2_x0_25_pretrained  -o Global.save_inference_dir=./output/
```

