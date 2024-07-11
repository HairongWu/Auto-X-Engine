# Auto-X Engine

Auto-X Engine is an open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios.

Auto-X Engine Server is an open source inference serving software that streamlines AI inferencing. It enables users to deploy AI models from Auto-X Engine. It supports inference across cloud, data center, edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. It also delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming.

The Auto-X Engine Core is a deep learning compute library designed for MCUs/CPUs. It is written entirely in C and provides elementary compute operator for other parts of this repo. Most of the codes are adapted from [ggml](https://github.com/ggerganov/ggml), [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite),
[OpenCV](https://github.com/opencv/opencv), [onnx2c](https://github.com/kraiskil/onnx2c) and [llama2.c](https://github.com/karpathy/llama2.c).

The Auto-X Engine AIOS is a RTOS system based on NuttX with native AI supported by Auto-X Engine.

This engine only supports the model structures described in this repo at this time. And these models are needed in the built-in solutions.



## Model Pool

> **Note** The following models could be modified from the originial ones.
> We also provide guidelines and running code to customize and retrain the following models using your own data.

### Tiny Models for MCU (such as ESP32 and Arm Cortex-M) (Lite)

- [picodet](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/picodet/README_en.md)
- [tinypose](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/keypoint/tiny_pose/README_en.md)
- [PaddleOCR Mobile](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)

### Medium Models for single CPU (such as Arm Cortex-A and X86) (Lite)

1. Llama2 & Llama3
   
The demo resides in the 'demos' folder with a MSVS project. As to the model downloading and other details, please refer to the following table:

| model | dim | n_layers | n_heads | n_kv_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 260K | 64 | 5 | 8 | 4 | 512 | 260K | 1.297 | [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)
| OG | 288 | 6 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |
  
2. Whisper

Please refer to [here](https://github.com/ggerganov/whisper.cpp) for details.

| Model  | Disk    | Mem     |
| ------ | ------- | ------- |
| tiny   | 75 MiB  | ~273 MB |
| base   | 142 MiB | ~388 MB |
| small  | 466 MiB | ~852 MB |

### Large-scale Models for CPUs/GPUs (Server)



## Reference

- [Triton Inference Server](https://github.com/triton-inference-server/server?tab=readme-ov-file)
- [MNN](https://github.com/alibaba/MNN)
- [onnx2c](https://github.com/kraiskil/onnx2c)
- [Apache NuttX](https://github.com/apache/nuttx)