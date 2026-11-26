# TinySegment

TinySegment turns a text prompt and an image into object masks using CLIP, YOLO-World, and EfficientViT-SAM. Each stage runs independently, so you can replace one model and try it in the same pipeline.

## Quickstart

**Set up the three vision baselines.** Run these commands from the repo root on macOS arm64.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python models/yolo_world_v2_small/baseline/restore_source.py
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
tinysegment setup-weights
```

**Run each stage** on the bundled bus image. CLIP encodes `person`, YOLO-World finds matching boxes, and SAM turns those boxes into masks.

```sh
tinysegment encode-text person --output outputs/text.npz
tinysegment detect tests/fixtures/bus.jpg --embeddings outputs/text.npz --output outputs/boxes.json
tinysegment segment tests/fixtures/bus.jpg --boxes outputs/boxes.json --output outputs/masks.npz
```

**Run the full pipeline** to save an overlay of the people in the image.

```sh
tinysegment run tests/fixtures/bus.jpg person --output outputs/overlay.png
```

**Try the experimental CPU segmenter.** Build `int8_stage3_stage4` locally from baseline SAM, then run it on the same image. Use the separate Python 3.12 environment from the [ExecuTorch setup and build recipe](models/efficientvit_sam_l0/int8_stage3_stage4/README.md).

```sh
python -m models.efficientvit_sam_l0.int8_stage3_stage4.build
tinysegment run tests/fixtures/bus.jpg person --segmenter-variant int8_stage3_stage4 --output outputs/int8.png
```

**Use the voice-controlled camera.** In the baseline environment, install Moonshine and start with `person`. Live speech appears in white at the bottom left. The current target stays pinned in cyan at the bottom right. Say “Reachy, bus” to switch targets. Press `q` to quit.

```sh
python -m pip install --no-build-isolation -e '.[speech]'
tinysegment setup-weights --speech
tinysegment video person
```

## Architecture

| Location | What lives here |
| --- | --- |
| `src/tinysegment/` | CLI, model selection, image pipeline, and live camera loop |
| `models/<family>/baseline/` | Original model source or runtime adapter, loader, and download manifest |
| `models/<family>/<variant>/` | Experimental model changes, build recipe, and loader |

Downloaded weights and built programs live in each baseline or variant’s Git-ignored `artifacts/` directory.

## How to edit

**Change the pipeline.** Edit `segment_image()` to change how detector boxes reach SAM, or `run()` to change the live camera loop. Both are in [pipeline.py](src/tinysegment/pipeline.py).

**Add a SAM variant.**

1. Create a directory beside `models/efficientvit_sam_l0/baseline/` with your changes, build recipe, and loader. Follow the [INT8 SAM example](models/efficientvit_sam_l0/int8_stage3_stage4/).
2. Have the loader return a segmenter with the baseline’s `segment(image, boxes)` interface. Register the loader by name in `pipeline.py` under `VARIANTS["segmenter"]`.
3. Pass `--variant NAME` to `tinysegment segment` or `--segmenter-variant NAME` to `tinysegment run`. Compare the outputs on the same image. See the [baseline validation notes](docs/validation.md).
