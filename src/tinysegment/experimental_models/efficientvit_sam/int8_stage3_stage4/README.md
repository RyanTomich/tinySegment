# EfficientViT-SAM L0 with INT8 stage 3 and 4 entries

This CPU variant loads the baseline SAM checkpoint and replaces the first block in each of backbone stages 3 and 4 with a compiled XNNPACK program. Each program accepts and returns FP32 tensors. The adapter exposes the same box-prompted segmentation API as baseline SAM.

## Build from the checkout

Use a separate Python 3.12 environment. The following commands use the validated macOS arm64 dependencies. Set the COCO path to a val2017 dataset directory containing the `val2017/` image folder. The build uses 16 fixed calibration images and downloads missing JPEGs into that folder.

```sh
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install executorch==1.6.0 torchao==0.18.0
tinysegment setup-weights
python -m tinysegment.experimental_models.efficientvit_sam.int8_stage3_stage4.build --coco-root /datasets/coco
```

The build prints a new directory under `runs/sam/` containing `stage3-int8.pte` and `stage4-int8.pte`. The loader defaults to `runs/sam/research-model/` in a checkout. To select a specific build, pass the printed directory.

```sh
tinysegment run vendor/ultralytics/assets/bus.jpg person --segmenter-variant int8_stage3_stage4 --segmenter-program-dir "$PROGRAM_DIR" --output outputs/int8.png
```

```python
from tinysegment.selection import load_stage

sam = load_stage("segmenter", "int8_stage3_stage4", program_dir=program_dir)
result = sam.segment(image, boxes)
```

With an installed wheel, supply `program_dir` and `weights_dir` explicitly. A checkpoint override selects the underlying SAM checkpoint. The loader records the path and SHA-256 hash of each compiled program in `sam.metadata`.
