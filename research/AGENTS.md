# SAM Research Guidelines

## Create a checkpoint

Run these commands from the repository root with the editable TinySegment installation described in the [root guide](../AGENTS.md).

```sh
source .venv/bin/activate
python -m pip install -r research/sam/requirements-benchmark.txt
tinysegment setup-weights
python -m research.sam.run --name baseline --recipe baseline
```

The `baseline` recipe saves a copy of the original EfficientViT-SAM L0 model for checking reproducibility. To test a change, register a recipe in `research/sam/recipes.py` and pass its key to `--recipe`. Use `--name` to label the run. Let the runner save the checkpoint.

## Find the run

Copy the directory printed after `Run directory:` into `RUN`. Each invocation gets a unique directory, even when names repeat. To find an earlier run and verify your selection, run the commands below.

```sh
ls -dt runs/sam/*/
RUN='runs/sam/<timestamp>-baseline-<suffix>'  # Replace with the actual directory.
cat "$RUN/config.json"
cat "$RUN/metadata.json"
```

Check the recipe in `config.json` and confirm generation has `status` set to `success` in `metadata.json`. The candidate is `$RUN/checkpoint.pt`. Each benchmark compares it with the verified original recorded in that run.

## Run the three benchmarks

```sh
python -m research.sam.benchmark quality "$RUN" --coco-root /path/to/coco
python -m research.sam.benchmark size-ops "$RUN"
python -m research.sam.benchmark latency "$RUN"
```

Run any command independently. Quality requires `val2017/` images and `annotations/instances_val2017.json` beneath the COCO root. It uses the fixed 100-image subset by default. Add `--full` for the complete validation split. Size and latency can run without COCO data.

## Find and interpret results

Each command prints its report path. To list saved reports, run `find "$RUN/evaluations" -name report.json`. Open the chosen file, check its `benchmark` and `status`, then read `metrics`. Deltas are candidate minus original.

- **Quality** — Under `groups.all`, positive `delta_points` means better mean mask IoU against human annotations. `fidelity_miou_pct` measures agreement with the original model. A no-op should give zero accuracy change and 100% fidelity.
- **Size and operations** — Under `comparison`, compare parameter counts, `tensor_payload_mib`, `checkpoint_mib`, and `total_gmac`. Negative deltas mean fewer parameters, bytes, or operations. Serialization can change file size with identical tensors. If `complete` is false, inspect `coverage` before interpreting MAC totals.
- **Latency** — Compare `median_ms` and `p95_ms` for `first_mask` (includes image encoding) and `cached_box` (reuses the embedding). Negative deltas mean faster inference. Repeat on the same idle machine to judge small changes against timing variation.
