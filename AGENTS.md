# Repository guidelines

TinySegment turns a prompt and an image into masks. To debug a prediction, save the output of the stage before the one you are changing. Detector experiments can reuse CLIP embeddings, and SAM experiments can reuse detector boxes. The same stage objects serve the image CLI and live camera.

## Set up a checkout

Run these commands from the repository root. The baseline lock is validated on macOS arm64 with Python 3.12 and CPU inference.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python models/yolo_world_v2_small/baseline/restore_source.py
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
tinysegment setup-weights
```

The restore script verifies the pinned Ultralytics source, downloading it if missing. Run it **before installing or building a wheel** so setuptools discovers the detector package. Keep detector modifications in tracked variant code or patches. The restored tree is Git-ignored. Preserve the upstream import names (`clip`, `ultralytics`, `efficientvit`, `segment_anything`) when changing packaging. The YOLO checkpoint deserializes an upstream Python class.

`models/storage.py` reads each family's `baseline/manifest.json`. Weight setup fetches missing files, verifies SHA-256, and makes the originals read-only. The editable checkout uses `models/` as its storage root. A wheel installation defaults to `~/.cache/tinysegment/models/`. To share a different root across setup, builds, and inference, set `TINYSEGMENT_WEIGHTS_DIR`. CLI commands also accept `--weights-dir DIR`. Supply the parent of the family directories. Write predictions under `outputs/`, outside weights storage and at paths distinct from all inputs.

## 1. Run the pipeline on an image

```sh
tinysegment run tests/fixtures/bus.jpg person --output outputs/overlay.png
```

Quote a multiword prompt. The CLI loads the models once, applies EXIF orientation, converts the image to RGB, and encodes the prompt. The detector finds boxes and SAM segments the clean image in detection order. The command prints the detection count and writes a full-resolution RGBA overlay. With zero detections the overlay shows the original image. To investigate a missed detection, inspect the boxes in step 2. Detector options include `--confidence` (default 0.25), `--iou` (0.7), and `--image-size` (640, a positive multiple of 32).

| Where to make a change | Responsibility |
| --- | --- |
| `src/tinysegment/cli.py` | CLI arguments, image orientation, stage file formats, output path checks, and still-image command dispatch |
| `src/tinysegment/pipeline.py` | `VARIANTS` and `load_stage()` select loaders. `load_models()` constructs the three stages. `segment_image()` passes detector boxes to SAM. `render_overlay()` draws the result. |
| `pipeline.run()` and `src/tinysegment/captions.py` | The live camera loop calls the same `segment_image()`. Moonshine supplies spoken prompts and the captions module draws preview text. |
| `models/<family>/baseline/loader.py` | Model construction, preprocessing, and inference for that family. Vision source, licenses, and provenance live under `baseline/upstream/`. |
| `models/stage.py` | Common vision-stage fields (`model`, `checkpoint`, `metadata`), evaluation mode, and device selection |
| `src/tinysegment/__init__.py` | Public Python imports for the three vision baseline loaders. Preserve these when moving model code. |

For a binary mask PNG, use `tinysegment run tests/fixtures/bus.jpg --stream --output outputs/mask.png` and send prompts one per line on stdin. It processes the latest pending prompt and replaces that PNG. For the optional camera path, install `python -m pip install --no-build-isolation -e '.[speech]'`, run `tinysegment setup-weights --speech`, then `tinysegment video person`. Its speech adapter lives in `models/moonshine_tiny_streaming_en/baseline/`.

## 2. Run a baseline model independently

Each command loads only the stage it exercises. These commands leave reusable inputs for testing a replacement on the same fixture.

```sh
tinysegment encode-text person --output outputs/text.npz
tinysegment detect tests/fixtures/bus.jpg --embeddings outputs/text.npz --output outputs/boxes.json
tinysegment segment tests/fixtures/bus.jpg --boxes outputs/boxes.json --output outputs/baseline.npz
```

| Stage key and model family | Python interface and file contract |
| --- | --- |
| `clip` · `clip_vit_b32` | `encode(prompts)` returns a normalized float tensor. The NPZ holds `embeddings` of shape `(1, number_of_labels, 512)` and ordered `labels`. Pass multiple labels as separate arguments to `encode-text`. |
| `detector` · `yolo_world_v2_small` | `detect(image, embeddings, labels, confidence, iou, image_size)` returns JSON-compatible `boxes`, `scores`, `class_ids`, `labels`, and `image_size` as `[width, height]`. Class IDs index the input labels. Boxes are pixel-space `[x1, y1, x2, y2]` on the EXIF-corrected image. |
| `segmenter` · `efficientvit_sam_l0` | `segment(image, boxes)` returns boolean `masks` of shape `(N, height, width)`, float32 `scores` of shape `(N,)`, and `combined_mask`, the union of all masks at `(height, width)`. The CLI saves these arrays in an NPZ. Empty boxes yield zero masks and an all-false union. |

To run SAM alone, you can also supply your own JSON, e.g. `{"image_size":[810,1080],"boxes":[[40,390,240,905]]}` for the bundled bus image. Use finite coordinates with positive box width and height. The CLI checks `image_size` against the oriented input when present.

Use `tinysegment inspect segmenter` (also `clip` or `detector`) to see the module tree, defining source file, checkpoint hash, and parameter count. In Python, `from tinysegment import load_text_encoder, load_detector, load_segmenter` gives access to the resident stages and their editable `.model` objects.

## 3. Make an experimental model

1. **Choose how to load the change.** For weights compatible with the existing loader and architecture, save a separate checkpoint. For example, SAM accepts `--checkpoint PATH` on `segment` and `--segmenter-checkpoint PATH` on `run`. For a different architecture, preprocessing path, or runtime, give the variant a package under `models/<family>/<variant>/` beside `baseline/`. Include `__init__.py` for setuptools discovery. If the loader needs a new tracked data file, check the package-data patterns in `pyproject.toml` so the file ships in the wheel.
2. **Implement the loader.** Put the changed modules and their construction in the variant package. In `loader.py`, accept the keyword arguments `weights_dir=None`, `checkpoint=None`, and `device='cpu'` that `load_stage()` passes. If modifying the baseline in memory, delegate to its loader with those arguments, replace the relevant modules, and return the stage in evaluation mode. Preserve the stage's input/output contract above and its `.model`, `.checkpoint`, and `.metadata` for `inspect`.
3. **Make it reproducible.** Keep build or export code, required dependencies, and a recipe beside the loader. Store generated checkpoints and programs in the variant's ignored `artifacts/`. Record the identity of inputs and compiled programs. Put model-specific fixtures and comparisons in the family's `evaluation/` when adding them. Shared fixtures belong in `tests/fixtures/`. Keep independent experiment snapshots and results in ignored `runs/`. Inference, builds, setup, and evaluations must work with an empty run history.

The implemented example is `models/efficientvit_sam_l0/int8_stage3_stage4/`. Its `build.py` captures inputs to the first block of backbone stages 3 and 4 using images pinned in `calibration.json`. Its `export.py` quantizes and lowers each block to XNNPACK. The builder publishes both programs together. At load time, `loader.py` starts from baseline SAM, replaces only `stages[index].op_list[0]` for those two stages, and records the program hashes in stage metadata. The existing predictor and mask decoder still handle boxes.

## 4. Build and run the experimental segmenter

Use a separate Python 3.12 environment for ExecuTorch. From the repo root, the following creates one under ignored `.runtime/`. The baseline setup above supplies the restored detector source and checkpoints.

```sh
python3.12 -m venv .runtime/stage4-env
source .runtime/stage4-env/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install executorch==1.6.0 torchao==0.18.0
python -m models.efficientvit_sam_l0.int8_stage3_stage4.build
```

The build verifies 16 calibration JPEGs and fetches any missing ones. To reuse local images, append `--coco-root DIR` where `DIR/val2017/` contains the JPEGs. On success the builder prints the directory containing `stage3-int8.pte` and `stage4-int8.pte`. The loader expects both under `efficientvit_sam_l0/int8_stage3_stage4/artifacts/` in the active weights root. For a custom location, pass the directory containing both files to `--segmenter-program-dir`. A checkpoint override selects the underlying SAM weights. [Variant build recipe](models/efficientvit_sam_l0/int8_stage3_stage4/README.md)

In that environment, reuse the image and boxes from step 2. This tests the CPU segmenter independently of CLIP and YOLO.

```sh
tinysegment segment tests/fixtures/bus.jpg --boxes outputs/boxes.json \
  --variant int8_stage3_stage4 --output outputs/int8.npz
```

To test a new loader before registering it, import it directly and compare masks for the saved boxes. This example uses the existing INT8 loader and reports one mask IoU per detection on the bus fixture.

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from models.efficientvit_sam_l0.int8_stage3_stage4.loader import load_research_segmenter

boxes = json.loads(Path("outputs/boxes.json").read_text())["boxes"]
with Image.open("tests/fixtures/bus.jpg") as source:
    image = ImageOps.exif_transpose(source).convert("RGB")
sam = load_research_segmenter(device="cpu")
result = sam.segment(image, boxes)
assert result["masks"].shape == (len(boxes), image.height, image.width)
with np.load("outputs/baseline.npz") as baseline:
    for ref, candidate in zip(baseline["masks"], result["masks"], strict=True):
        union = (ref | candidate).sum()
        print((ref & candidate).sum() / union if union else 1.0)
```

## 5. Integrate a variant into the pipeline

1. In `src/tinysegment/pipeline.py`, map the variant name to its loader under the matching stage in `VARIANTS`. Follow `_load_int8` to defer optional imports until selection. Keep `baseline` as the default for each stage.
2. The CLI reads this registry for `run` and `video`. Standalone commands and `inspect` pass `--variant NAME` to `load_stage()`. Full-pipeline flags are `--clip-variant`, `--detector-variant`, and `--segmenter-variant`. If a new variant needs an explicit program directory, update the INT8-specific check in `load_stage()` and the program input paths passed to `validate_output_path()` in `cli.py`.
3. Exercise the standalone stage first, then select it in the full pipeline. With the supplied INT8 variant this leaves CLIP and YOLO on their baselines.

```sh
tinysegment run tests/fixtures/bus.jpg person \
  --segmenter-variant int8_stage3_stage4 --output outputs/int8.png
```

Compare `outputs/baseline.npz` and `outputs/int8.npz` by box index before interpreting the overlays. Check shape, mask IoU, and scores with identical boxes. Agreement with the baseline measures drift. For a claim about accuracy, compare both against the same ground truth. For a performance claim, measure the relevant size, operation count, or latency using the same inputs, device, and thread count. [Baseline validation and fixture tolerances](docs/validation.md)

## Checks and contribution conventions

Run these from the baseline environment (`source .venv/bin/activate`).

```sh
python -m unittest discover -s tests
TINYSEGMENT_INTEGRATION=1 python -m unittest discover -s tests
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

- `tests/test_baseline.py` checks stage contracts and EXIF orientation. With weights installed, its integration tests run each CLI stage and the full pipeline in isolation with network and run-history access blocked.
- `tests/test_stage4.py` shows how to test variant selection, missing programs, device restrictions, and the exact modules replaced. `tests/test_weights.py` covers checksums and output path protection. The stream, speech, and video tests cover the shared live paths.
- Place `unittest` cases in `tests/test_*.py` and name methods `test_*`. Follow the surrounding four-space Python style, with `snake_case` for functions and modules and `PascalCase` for classes. There is no configured formatter or linter.
- Use a short imperative commit summary. In a pull request, name the affected stage and behavior, list checks run, link the relevant issue when available, and include before-and-after measurements for accuracy or performance claims.
