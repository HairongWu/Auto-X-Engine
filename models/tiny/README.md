# Tiny Models

## ShuffleNetV2

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
python tools/export_ppcls.py -c ./configs/ShuffleNetV2_x0_25.yaml -o Global.pretrained_model=./pretrain/ShuffleNetV2_x0_25_pretrained  -o Global.save_inference_dir=./output/ShuffleNetV2_x0_25/
```

## PP-PicoDet

### Deployment

```shell
>python tools/export_ppdet.py -c configs/picodet/picodet_xs_320_coco_lcnet.yml -o weights=./pretrain/picodet_xs_320_coco_lcnet.pdparams --output_dir ./output/picodet_xs_320_coco_lcnet
```

## PP-TinyPose


### Deployment

```bash
python tools/export_ppdet.py -c configs/keypoint/tiny_pose/tinypose_128x96.yml -o weights=pretrain/tinypose_128x96 --output_dir ./output/tinypose_128x96
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
### Deployment

```bash
python tools/export_ppocr.py -c configs/japan_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain/japan_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=./output/japan_PP-OCRv3_rec/
```

## MobileSeg

### Deployment

```bash
tools/export_paddleseg.py --config ./configs/pp_mobileseg/pp_mobileseg_tiny_ade20k_512x512_80k.yml --model_path pretrain/pp_mobileseg_tiny_ade20k_512x512_80k.pdparams --input_shape 1 3 512 512
```