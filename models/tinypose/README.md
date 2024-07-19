# PP-TinyPose

PP-TinyPose is a real-time keypoint detection model optimized by PaddleDetecion for mobile devices, which can smoothly run multi-person pose estimation tasks on mobile devices. 

PP-TinyPose has the following dependency requirements:
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)>=2.2

| Model  | Input Size | AP (COCO Val) | Inference Time for Single Person (FP32)| Inference Time for Single Person（FP16) | Config | Model Weights | Deployment Model | Paddle-Lite Model（FP32) | Paddle-Lite Model（FP16)|
| :------------------------ | :-------:  | :------: | :------: |:---: | :---: | :---: | :---: | :---: | :---: |
| PP-TinyPose | 128*96 | 58.1 | 4.57ms | 3.27ms | [Config](./tinypose_128x96.yml) |[Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_fp16_lite.tar) |
| PP-TinyPose | 256*192 | 68.8 | 14.07ms | 8.33ms | [Config](./tinypose_256x192.yml) | [Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_fp16_lite.tar) |

**Tips**
- The AP results of keypoint detection models are based on bounding boxes detected by corresponding detection model.
- In accuracy evaluation, there is no flip, and threshold of bounding boxes is set to 0.5.
- For fairness, in multi-persons test, we remove images with more than 6 people.
- The inference time is tested on a Qualcomm Snapdragon 865, with 4 threads at arm8, FP32.
- Pipeline time includes time for preprocess, inferece and postprocess.
- About the deployment and testing for other opensource model, please refer to [Here](https://github.com/zhiboniu/MoveNet-PaddleLite).
- For more performance data in other runtime environment, please refer to [Keypoint Inference Benchmark](../KeypointBenchmark.md).

## Model Training
In addition to `COCO`, the trainset for keypoint detection model and pedestrian detection model also includes [AI Challenger](https://arxiv.org/abs/1711.06475). Keypoints of each dataset are defined as follows:
```
COCO keypoint Description:
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder,
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"

AI Challenger Description:
    0: "Right Shoulder",
    1: "Right Elbow",
    2: "Right Wrist",
    3: "Left Shoulder",
    4: "Left Elbow",
    5: "Left Wrist",
    6: "Right Hip",
    7: "Right Knee",
    8: "Right Ankle",
    9: "Left Hip",
    10: "Left Knee",
    11: "Left Ankle",
    12: "Head top",
    13: "Neck"
```

Since the annatation format of these two datasets are different, we aligned their annotations to `COCO` format. You can download [Training List](https://bj.bcebos.com/v1/paddledet/data/keypoint/aic_coco_train_cocoformat.json) and put it at `dataset/`. To align these two datasets, we mainly did the following works:
- Align the indexes of the `AI Challenger` keypoint to be consistent with `COCO` and unify the flags whether the keypoint is labeled/visible.
- Discard the unique keypoints in `AI Challenger`. For keypoints not in this dataset but in `COCO`, set it to not labeled.
- Rearranged `image_id` and `annotation id`.

Training with merged annotation file converted to `COCO` format:
```bash
# keypoint detection model
python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/tiny_pose/tinypose_128x96.yml

# pedestrian detection model
python3 -m paddle.distributed.launch tools/train.py -c configs/picodet/application/pedestrian_detection/picodet_s_320_pedestrian.yml
```

## Model Deployment

### Export the trained model through the following command:
```bash
python3 tools/export_model.py -c configs/picodet/application/pedestrian_detection/picodet_s_192_pedestrian.yml --output_dir=outut_inference -o weights=output/picodet_s_192_pedestrian/model_final

python3 tools/export_model.py -c configs/keypoint/tiny_pose/tinypose_128x96.yml --output_dir=outut_inference -o weights=output/tinypose_128x96/model_final
```

### Migrate to ESP32