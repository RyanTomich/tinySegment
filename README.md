# TinySegment

Editable, standalone PyTorch inference for **CLIP ViT-B/32 → YOLO-World v2 small → EfficientViT-SAM L0**. Each stage runs separately and exposes its model architecture and weights. No other checkout is required.

## Install

Python 3.12 is the tested baseline. From this checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/tinysegment setup-weights
```

The lock records the tested macOS arm64 environment. On other platforms, install with `pip install -e .` in a fresh environment to resolve platform-compatible dependencies; those platforms are not yet validated. Do not separately install `ultralytics`, `clip`, `segment-anything`, or `efficientvit`: their editable sources are included here.

The three checkpoints are already present in this working copy under `weights/` (approximately 496 MB). They are ignored by Git. `setup-weights` verifies existing files and explicitly downloads missing ones, checking SHA-256 against the [manifest](src/tinysegment/weights.json). Inference never downloads checkpoints. Use only trusted checkpoint files: the original YOLO checkpoint contains a serialized Python module.

Defaults resolve relative to this checkout, regardless of the current working directory. Override with `--weights-dir PATH`, `TINYSEGMENT_WEIGHTS_DIR`, or `--checkpoint FILE` on individual commands. A non-editable installation defaults to `~/.cache/tinysegment/weights`. `run` accepts `--clip-checkpoint`, `--detector-checkpoint`, and `--segmenter-checkpoint` separately.

## Run each stage

The bundled Ultralytics bus image is a convenient smoke-test input:

```sh
.venv/bin/tinysegment encode-text person bus --output outputs/text.npz
.venv/bin/tinysegment detect vendor/ultralytics/assets/bus.jpg \
  --embeddings outputs/text.npz --output outputs/boxes.json
.venv/bin/tinysegment segment vendor/ultralytics/assets/bus.jpg \
  --boxes outputs/boxes.json --output outputs/masks.npz
.venv/bin/tinysegment run vendor/ultralytics/assets/bus.jpg person \
  --output outputs/overlay.png
.venv/bin/tinysegment inspect detector
```

Use `--help` on any command. CPU FP32 is the default; `--device mps` or `--device cuda:0` opts into another available backend. Only CPU has been validated. Detection defaults to image size 640, confidence 0.25, and NMS IoU 0.7. Image size must be a multiple of 32.

`run` accepts one text prompt, matching the original demo. `encode-text` accepts multiple ordered prompts. SAM receives the original EXIF-corrected RGB image; green markers are drawn only on the output. `--legacy-marked-input` reproduces the old marker-before-segmentation behavior. No detections produces the unmarked RGBA image.

Artifacts:

- Text `.npz`: `embeddings` float32 `[1, N, 512]`, unit-normalized; `labels` ordered strings. Long prompts use CLIP's 77-token truncation.
- Detection JSON: `boxes` pixel-coordinate `[x1,y1,x2,y2]`, `scores`, `class_ids`, `labels`, and `image_size` `[width,height]`. Coordinates refer to the EXIF-corrected image.
- Mask `.npz`: `masks` bool `[N,H,W]`, `scores` float32 `[N]` (SAM mask-quality estimates), and `combined_mask` bool `[H,W]`. Box order is preserved. You can supply your own JSON containing `boxes` and optional `image_size`.

## Edit architectures and weights

```python
from tinysegment import load_text_encoder, load_detector, load_segmenter
from tinysegment.pipeline import read_image

text = load_text_encoder()
detector = load_detector()
sam = load_segmenter()
image = read_image('vendor/ultralytics/assets/bus.jpg')
features = text.encode(['person', 'bus'])
detections = detector.detect(image, features, ['person', 'bus'])
masks = sam.segment(image, detections['boxes'])

print(detector.model)                         # eager WorldModel, not fused
print(detector.checkpoint)                    # local checkpoint path
weights = detector.model.state_dict()
for name, layer in detector.model.named_modules():
    print(name, type(layer).__name__)
# Standard PyTorch replacement and state_dict APIs work on every .model.
image_encoder = sam.model.image_encoder
prompt_encoder = sam.model.prompt_encoder
mask_decoder = sam.model.mask_decoder
```

For tensor-level experiments, use `text.tokenize` + `text.forward`, `detector.preprocess` + `detector.forward` + `detector.postprocess`, and SAM's exposed encoder/decoder modules. Convenience inference methods disable gradients; raw model modules remain editable. `sam.predictor.set_image` caches image features and `predict` reuses them across boxes. Reset/recompute features after changing the image encoder. Detection updates the class count per call, so adapters should not be shared across concurrent calls with different prompts.

Source locations:

| Component | Editable implementation |
|---|---|
| TinySegment adapters and CLI | `src/tinysegment/` |
| CLIP text/vision architecture and tokenizer | `vendor/clip/` |
| YOLO architecture, layers, configs | `vendor/ultralytics/nn/`, `vendor/ultralytics/cfg/models/v8/yolov8-worldv2.yaml` |
| EfficientViT image encoder and SAM integration | `vendor/efficientvit/models/efficientvit/` |
| SAM prompt encoder and mask decoder | `vendor/segment_anything/modeling/` |

The initial checkpoint-loading paths use upstream formats. For custom CLIP/SAM checkpoints, save compatible state dictionaries. For YOLO, load a baseline model and call its `load_state_dict()` from Python; `--checkpoint` expects the upstream module-containing format. Architecture changes may require matching loader changes.

## Validate

```sh
.venv/bin/python -m unittest discover -s tests -v
TINYSEGMENT_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -v
```

Integration tests use real weights, run commands from a temporary directory, reject network and sibling-project access, check stage isolation, and round-trip model state dictionaries. See [validation results](docs/validation.md).

Quantization, model export, training, and a full performance benchmark harness are future work. The EfficientViT copy is an inference subset; upstream training helpers such as `reset_bn` that require its training package are outside this baseline.

## Third-party sources

[Provenance](vendor/provenance.json), [source hashes](vendor/source-hashes.json), and upstream license notices in `vendor/licenses/` document the imported baseline. EfficientViT's Apache license is also retained in its package. Ultralytics and its CLIP fork carry AGPL notices; retain the upstream licenses when redistributing these sources. Model packages keep their original import names for checkpoint compatibility. No virtual environment or external checkout is vendored.

## SAM experiments and checkpoint protection

The [standard research SAM](research/sam/README.md#standard-research-sam) runs the stage 3 and stage 4 entry blocks with XNNPACK INT8. Use `research.sam.model.load_research_segmenter()` to load it and `tinysegment.load_segmenter()` for original FP32 SAM.

Research code is separate from inference in [`research/sam/`](research/sam/README.md). Start with:

```sh
.venv/bin/python -m research.sam.inspect
.venv/bin/python -m research.sam.run --name baseline
.venv/bin/python -m research.sam.evaluate runs/sam/<run-id>
```

Generation prints the run directory containing the checkpoint, configuration, provenance, and architecture inspection. Pass that directory to the evaluator to compare the saved model with the original. Each invocation writes its own `evaluations/<id>/report.json` inside the run. Select an evaluated checkpoint explicitly with `tinysegment run --segmenter-checkpoint ...` or `tinysegment segment --checkpoint ...`. Default inference continues using the original weights.

Original checkpoint hashes are verified when loaded, and `setup-weights` marks originals read-only. Research writes never target original weights; existing run artifacts cannot be replaced. Inference output guards also reject checkpoint destinations. See the [research guide](research/sam/README.md) for adding recipes, chaining experiments, and interpreting baseline comparisons.
