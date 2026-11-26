# EfficientViT-SAM L0 with INT8 stage 3 and 4 entries

This CPU variant loads the baseline SAM checkpoint and replaces the first block in each of backbone stages 3 and 4 with a compiled XNNPACK program. Each program accepts and returns FP32 tensors. The adapter exposes the same box-prompted segmentation API as baseline SAM.

## Build from the checkout

Use a separate Python 3.12 environment. The following commands use the validated macOS arm64 dependencies. The build retrieves 16 calibration JPEGs into the local cache and verifies their SHA-256 hashes. Pass `--coco-root DIR` to reuse `DIR/val2017/`.

```sh
python models/yolo_world_v2_small/baseline/restore_source.py
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install executorch==1.6.0 torchao==0.18.0
tinysegment setup-weights
python -m models.efficientvit_sam_l0.int8_stage3_stage4.build
```

The build prints the variant’s local `artifacts/` directory after both exports succeed. The loader discovers both programs there. If you keep an older build in a custom weights directory, move its `int8_stage3_stage4/` folder beside `baseline/` or pass its existing `artifacts/` path to `--segmenter-program-dir`.

```sh
tinysegment run tests/fixtures/bus.jpg person --segmenter-variant int8_stage3_stage4 --output outputs/int8.png
```

```python
from tinysegment.pipeline import load_stage

sam = load_stage("segmenter", "int8_stage3_stage4")
result = sam.segment(image, boxes)
```

A checkpoint override selects the underlying SAM checkpoint. The loader records the path and SHA-256 hash of each compiled program in `sam.metadata`.

## Recorded comparison

On one Apple M4 Pro CPU thread, median latency from opening the bus JPEG to returning the first full-resolution mask was 407.9 ms for baseline SAM and 316.1 ms for this variant over 30 paired requests. On 742 COCO instances across 100 images, mean box-prompted mask IoU against ground truth was 78.41% for baseline and 77.96% for this variant.
