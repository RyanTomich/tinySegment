# Repository Guidelines

## Project structure

TinySegment runs three independent PyTorch stages. CLIP encodes text, YOLO-World detects boxes, and EfficientViT-SAM L0 segments each box. Keep baseline adapters in `src/tinysegment/baseline_models/` and selectable variants and their build recipes in `src/tinysegment/experimental_models/`. Resolve variants in `selection.py`, run inference in `pipeline.py`, and handle CLI arguments in `cli.py`. Keep `models.py` as compatibility imports.

`vendor/` contains the editable upstream implementations and assets, including the bus image used in smoke tests. Put historical SAM experiments, evaluation fixtures, and benchmarks in `research/sam/`. Keep original checkpoints in `weights/` and generated checkpoints and reports in `runs/sam/`. Both directories are Git-ignored.

## Development commands

Use Python 3.12. From the repository root, create and activate a virtual environment with `python3.12 -m venv .venv` and `source .venv/bin/activate`.

- On the validated macOS arm64 platform, install dependencies with `python -m pip install -r requirements.lock`, then install the editable checkout with `python -m pip install --no-build-isolation --no-deps -e .`.
- Run `tinysegment setup-weights` to fetch missing checkpoints and verify their hashes. Exercise the pipeline with `tinysegment run vendor/ultralytics/assets/bus.jpg person --output outputs/overlay.png`.
- To exercise the CPU INT8 variant in the ExecuTorch environment, run `tinysegment run vendor/ultralytics/assets/bus.jpg person --segmenter-variant int8_stage3_stage4 --output outputs/int8.png`.
- Build a wheel with `python -m pip wheel --no-deps --no-build-isolation . -w dist`.

## Coding style

Use four-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes. Follow the surrounding Python style. The project has no configured formatter or linter. Preserve box order and use pixel-space `xyxy` coordinates on the EXIF-corrected image when passing detections to SAM.

## Testing

Place `unittest` cases in `tests/test_*.py` and name test methods `test_*`. Run `python -m unittest discover -s tests` for routine checks. Install optional benchmark dependencies with `python -m pip install -r research/sam/requirements-benchmark.txt`. After setting up weights, use `TINYSEGMENT_INTEGRATION=1 python -m unittest discover -s tests` to include real-model tests. For SAM performance changes, report the relevant `quality`, `size-ops`, or `latency` comparison. See the [SAM research guide](research/sam/README.md).

## Commits and pull requests

Git history contains only `Initial commit`. Use a short, imperative summary such as `Add cached-box latency benchmark`. In a pull request, describe the affected stage and behavior, list the checks you ran, and link a relevant issue when available. Include before-and-after metrics for accuracy or performance claims.
