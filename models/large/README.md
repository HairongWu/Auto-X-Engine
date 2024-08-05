# Large Models

## SAM

The example currently supports only the [ViT-B SAM model checkpoint](https://huggingface.co/facebook/sam-vit-base).

```bash
# Download PTH model
wget -P ./pretrained/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Convert PTH model to ggml
python ./sam-pth-to-ggml.py ./pretrained/sam_vit_b_01ec64.pth ./pretrained/ 0
```

## References

- [ggml](https://github.com/ggerganov/ggml)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)