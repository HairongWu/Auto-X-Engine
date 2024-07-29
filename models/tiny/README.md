# Tiny Models

## ShuffleNetV2

The ShuffleNet series network is the lightweight network structure proposed by MEGVII. So far, there are two typical structures in this series network, namely, ShuffleNetV1 and ShuffleNetV2. A Channel Shuffle operation in ShuffleNet can exchange information between groups and perform end-to-end training. In the paper of ShuffleNetV2, the author proposes four criteria for designing lightweight networks, and designs the ShuffleNetV2 network according to the four criteria and the shortcomings of ShuffleNetV1.

### Benchmark

| Models                               | Top1    | Top5    | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ShuffleNetV2                         | 0.688   | 0.885   | 0.694             |                   | 0.280        | 2.260             |
| ShuffleNetV2_x0_25                   | 0.499   | 0.738   |                   |                   | 0.030        | 0.600             |
| ShuffleNetV2_x0_33                   | 0.537   | 0.771   |                   |                   | 0.040        | 0.640             |
| ShuffleNetV2_x0_5                    | 0.603   | 0.823   | 0.603             |                   | 0.080        | 1.360             |
| ShuffleNetV2_x1_5                    | 0.716   | 0.902   | 0.726             |                   | 0.580        | 3.470             |
| ShuffleNetV2_x2_0                    | 0.732   | 0.912   | 0.749             |                   | 1.120        | 7.320             |
| ShuffleNetV2_swish                   | 0.700   | 0.892   |                   |                   | 0.290        | 2.260             |

### Environment

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
pip install paddlelite
```

If you want to use PaddlePaddle on CPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
pip install paddlelite
```
### Deployment
Python3.7 is required.

```shell
python tools/export_shufflenetv2.py -c ./configs/ShuffleNetV2_x0_25.yaml -o Global.pretrained_model=./pretrain/ShuffleNetV2_x0_25_pretrained  -o Global.save_inference_dir=./output/ShuffleNetV2_x0_25/
```

## PP-PicoDet

`PP-PicoDet` are very suitable for deployment on mobile or CPU. For more details, please refer to our [report on arXiv](https://arxiv.org/abs/2111.00902).

- 🌟 Higher mAP: the **first** object detectors that surpass mAP(0.5:0.95) **30+** within 1M parameters when the input size is 416.
- 🚀 Faster latency: 150FPS on mobile ARM CPU.
- 😊 Deploy friendly: support PaddleLite/MNN/NCNN/OpenVINO and provide C++/Python/Android implementation.
- 😍 Advanced algorithm: use the most advanced algorithms and offer innovation, such as ESNet, CSP-PAN, SimOTA with VFL, etc.

### Benchmark

| Model     | Input size | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params<br><sup>(M) | FLOPS<br><sup>(G) | Latency<sup><small>[CPU](#latency)</small><sup><br><sup>(ms) | Latency<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |  Weight  | Config | Inference Model |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: | :----------------------------------------: | :--------------------------------------- | :--------------------------------------- |
| PicoDet-XS |  320*320   |          23.5           |        36.1       |        0.70        |       0.67        |              3.9ms              |            7.81ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_xs_320_coco_lcnet.yml) | [w/ postprocess](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_320_coco_lcnet.tar) &#124; [w/o postprocess](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_320_coco_lcnet_non_postprocess.tar) |
| PicoDet-XS |  416*416   |          26.2           |        39.3        |        0.70        |       1.13        |              6.1ms             |            12.38ms             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_xs_416_coco_lcnet.yml) | [w/ postprocess](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_416_coco_lcnet.tar) &#124; [w/o postprocess](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_xs_416_coco_lcnet_non_postprocess.tar) |


### Deployment

```shell
python tools/export_picodet.py -c configs/picodet/picodet_xs_320_coco_lcnet.yml -o weights=./pretrain/picodet_xs_320_coco_lcnet.pdparams 
```

- If no post processing is required, please specify: `-o export.post_process=False` (if -o has already appeared, delete -o here) or manually modify corresponding fields in [runtime.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/runtime.yml).
- If no NMS is required, please specify: `-o export.nms=True` or manually modify corresponding fields in [runtime.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/runtime.yml). Many scenes exported to ONNX only support single input and fixed shape output, so if exporting to ONNX, it is recommended not to export NMS.

## PP-TinyPose

PP-TinyPose is a real-time keypoint detection model optimized by PaddleDetecion for mobile devices, which can smoothly run multi-person pose estimation tasks on mobile devices.

### Keypoint Detection Model
| Model  | Input Size | AP (COCO Val) | Inference Time for Single Person (FP32)| Inference Time for Single Person（FP16) | Config | Model Weights | Deployment Model | Paddle-Lite Model（FP32) | Paddle-Lite Model（FP16)|
| :------------------------ | :-------:  | :------: | :------: |:---: | :---: | :---: | :---: | :---: | :---: |
| PP-TinyPose | 128*96 | 58.1 | 4.57ms | 3.27ms | [Config](./tinypose_128x96.yml) |[Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_fp16_lite.tar) |
| PP-TinyPose | 256*192 | 68.8 | 14.07ms | 8.33ms | [Config](./tinypose_256x192.yml) | [Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_fp16_lite.tar) |


### Deployment

```bash
python tools/export_tinypose.py -c configs/keypoint/tiny_pose/tinypose_128x96.yml -o weights=pretrain/tinypose_128x96
```

## PaddleOCR

### Environment

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
pip install Polygon3
pip install lanms-neo
pip install lmdb
pip install albumentations
```

### Benchmark

|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_rec_slim | [New] Slim quantization with distillation lightweight model, supporting Chinese, English text recognition |[ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 4.9M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.nb) |
|ch_PP-OCRv3_rec| [New] Original lightweight model, supporting Chinese, English, multilingual text recognition |[ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 12.4M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
|ch_PP-OCRv2_rec_slim| Slim quantization with distillation lightweight model, supporting Chinese, English text recognition|[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)| 9.0M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|ch_PP-OCRv2_rec| Original lightweight model, supporting Chinese, English, and multilingual text recognition |[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)|8.5M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
|ch_ppocr_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| 6.0M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
|ch_ppocr_mobile_v2.0_rec|Original lightweight model, supporting Chinese, English and number recognition|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|5.2M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|General model, supporting Chinese, English and number recognition|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |
|en_PP-OCRv3_rec_slim | [New] Slim quantization with distillation lightweight model, supporting English, English text recognition |[en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)| 3.2M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.nb) |
|en_PP-OCRv3_rec| [New] Original lightweight model, supporting English, English, multilingual text recognition |[en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)| 9.6M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
|en_number_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)| 2.7M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_train.tar) |
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition|[rec_en_number_lite_train.yml](../../configs/rec/multi_language/rec_en_number_lite_train.yml)|2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |
| korean_PP-OCRv3_rec | ppocr/utils/dict/korean_dict.txt |Lightweight model for Korean recognition|[korean_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml)|11.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| japan_PP-OCRv3_rec | ppocr/utils/dict/japan_dict.txt |Lightweight model for Japanese recognition|[japan_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/japan_PP-OCRv3_rec.yml)|11.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| chinese_cht_PP-OCRv3_rec | ppocr/utils/dict/chinese_cht_dict.txt | Lightweight model for chinese cht|[chinese_cht_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/chinese_cht_PP-OCRv3_rec.yml)|12.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar) |
| te_PP-OCRv3_rec | ppocr/utils/dict/te_dict.txt | Lightweight model for Telugu recognition |[te_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/te_PP-OCRv3_rec.yml)|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar) |
| ka_PP-OCRv3_rec | ppocr/utils/dict/ka_dict.txt | Lightweight model for Kannada recognition |[ka_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ka_PP-OCRv3_rec.yml)|9.9M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar) |
| ta_PP-OCRv3_rec | ppocr/utils/dict/ta_dict.txt |Lightweight model for Tamil recognition|[ta_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/ta_PP-OCRv3_rec.yml)|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar) |
| latin_PP-OCRv3_rec |  ppocr/utils/dict/latin_dict.txt | Lightweight model for latin recognition |  [latin_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/latin_PP-OCRv3_rec.yml) |9.7M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar) |
| arabic_PP-OCRv3_rec | ppocr/utils/dict/arabic_dict.txt | Lightweight model for arabic recognition  | [arabic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/arabic_PP-OCRv3_rec.yml) |9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar) |
| cyrillic_PP-OCRv3_rec | ppocr/utils/dict/cyrillic_dict.txt | Lightweight model for cyrillic recognition  | [cyrillic_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml) |9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar) |
| devanagari_PP-OCRv3_rec | ppocr/utils/dict/devanagari_dict.txt | Lightweight model for devanagari recognition | [devanagari_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml) |9.9M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar) |

### Deployment

```bash
python tools/export_ppocr.py -c configs/japan_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain/japan_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=./output/japan_PP-OCRv3_rec/
```