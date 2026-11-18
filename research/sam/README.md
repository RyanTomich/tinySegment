# SAM research workflow

This directory is for understanding and improving **EfficientViT-SAM L0**. Inference lives separately in `src/tinysegment`. Original checkpoints stay in `weights/`; generated experiment checkpoints and reports go into `runs/sam/`.

Run these commands from the TinySegment checkout with its environment activated:

```sh
source .venv/bin/activate
python -m research.sam.inspect
python -m research.sam.run --name baseline
```

Inspection prints the architecture, parameter counts by component, every parameter's name/shape/dtype, and the checkpoint hash. It does not dump the tensor values. For numerical inspection in Python:

```python
from tinysegment import load_segmenter
sam = load_segmenter()
parameters = dict(sam.model.named_parameters())
name = next(iter(parameters))
print(name, parameters[name].detach())
print(sam.model.image_encoder)
print(sam.model.prompt_encoder)
print(sam.model.mask_decoder)
```

## Generate a checkpoint

`python -m research.sam.run --name baseline` prints the new directory under `runs/sam/`. The runner verifies the original weights, loads the chosen input checkpoint, applies the recipe in memory, and saves a CPU state dictionary. Generation uses CPU FP32 and seed 0 by default (`--seed` overrides it).

The run contains `checkpoint.pt`, `inspection.txt` describing the transformed model, `config.json` with the recipe, seed, and checkpoint paths, and `metadata.json` with generation status, checkpoint hashes, dependency versions, and code provenance. A successful generation means the checkpoint was written and the input integrity checks passed. Evaluation checks whether the saved model loads and how closely it reproduces the original on a fixture.

## Evaluate a saved checkpoint

Use the run directory printed by generation. The evaluator loads the original checkpoint recorded in that run and compares it with `checkpoint.pt` using the normal inference loader. To evaluate another checkpoint in the same run, pass its filename with `--checkpoint`.

```sh
python -m research.sam.evaluate runs/sam/<run-id>
python -m research.sam.evaluate runs/sam/<run-id> --checkpoint another.pt \
  --image /path/to/image.jpg --boxes /path/to/boxes.json
```

The default fixture is the bundled bus image and two person boxes from `fixture.json`. For a custom fixture, supply both flags. Boxes JSON uses `{"boxes": [[x1, y1, x2, y2]], "image_size": [width, height]}`. Coordinates refer to the EXIF-corrected RGB image. Image size is optional. Supply at least one finite, positive-area box.

Each invocation prints its own `evaluations/<id>/report.json` under the run. The report identifies the evaluated files and records completion status. Successful reports include parameter counts, checkpoint sizes, pixel agreement, and per-box and mean mask IoU. These metrics measure fidelity to the original on the selected fixture. A failed or interrupted evaluation records its error when possible. A missing report indicates an incomplete evaluation. Existing runs from the combined workflow can be evaluated with the same command.

## Use a saved checkpoint

Copy the exact run directory printed by the runner; there is intentionally no automatic latest/best selection:

```sh
tinysegment run image.jpg person \
  --segmenter-checkpoint runs/sam/<run-id>/checkpoint.pt \
  --output outputs/experiment.png

tinysegment segment image.jpg --boxes boxes.json \
  --checkpoint runs/sam/<run-id>/checkpoint.pt \
  --output outputs/experiment-masks.npz
```

You can also load it directly with `load_segmenter(checkpoint="runs/sam/<run-id>/checkpoint.pt")`. Without an override, the pipeline always uses the original checkpoint.

To start a new experiment from a previous one:

```sh
python -m research.sam.run --name followup \
  --input-checkpoint runs/sam/<prior-run-id>/checkpoint.pt
```

## Add a model improvement

Add a function to `recipes.py` and register it in `RECIPES`. It receives `(model, config)` and returns the resulting model. Use `--recipe NAME` to select it. The initial `baseline` recipe is intentionally a no-op, so its saved checkpoint should match the original on the default fixture.

Do transformations on the in-memory model and let the runner save it. Do not call `torch.save` from recipes or modify files in `vendor/` or `weights/`. Future training recipes must explicitly switch to training mode; inference loading defaults to evaluation mode.

This version accepts compatible eager FP32 L0 state dictionaries. Generation rejects quantized tensors. Evaluation checks loader compatibility and compares the serialized model with the original. Recipe changes must be representable in the saved state dictionary to affect this comparison. Structural architecture changes and quantized runtimes need a matching, explicitly versioned inference-loader extension first.

## Original-weight protection

- Original checkpoint hashes are checked on default loading and at experiment start/end.
- `tinysegment setup-weights` verifies originals and makes them read-only. It refuses mismatches and never overwrites existing files. To restore a damaged baseline, deliberately remove the bad file and rerun setup.
- Research outputs have no arbitrary destination flag. Runs are created exclusively under `runs/sam`, symlinks are rejected, and artifacts are published atomically without replacing existing files. Published artifacts are also read-only.
- Inference commands reject outputs inside baseline storage, checkpoint filenames, and aliases of selected input files. Hard links and symlinks to known originals are protected.
- Original and generated weight files are excluded from Git. Keep a separate backup for storage failures; read-only files and application checks do not prevent deliberate permission changes, deletion, or writes by unrelated programs.

Inference does not import this research package. These scripts are checkout-local development tools and are not installed into the inference wheel.

## Benchmark a saved checkpoint

Install the optional research packages in the checkout environment. Each command accepts the run directory and an optional `--checkpoint filename.pt`. Each writes a new JSON report under that run's `evaluations/` directory with both checkpoint hashes, settings, and measurements.

```sh
python -m pip install -r research/sam/requirements-benchmark.txt
python -m research.sam.benchmark quality "$RUN" --coco-root /datasets/coco
python -m research.sam.benchmark size-ops "$RUN"
python -m research.sam.benchmark latency "$RUN"
```

- **Quality** reads `val2017/*.jpg` and `annotations/instances_val2017.json` under the supplied COCO root. The fixed manifest selects 100 images using Python's `random.Random(0).sample` on sorted eligible image IDs and includes all non-crowd annotations with a segmentation, positive area, and finite positive-size boxes. Add `--full` to score every eligible instance. Each model sees the same ground-truth box and returns one mask. The report averages per-instance IoU against COCO for accuracy and between models for fidelity. Scores use a 0–100 scale. Delta is candidate accuracy minus original accuracy in points. The size groups use annotation area in pixels with cutoffs of 32² and 96². Empty bins have a zero count and no score. [EfficientViT-SAM, Table 3](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.pdf)
- **Size and operations** reports parameter counts, checkpoint and tensor payload MiB, and dense MACs in billions. One multiply–accumulate counts as one MAC. The total covers the 512×512 image encoder plus the prompt encoder and mask decoder for one box. Bias additions, elementwise operations, normalization, resizing, and mask postprocessing are outside this count. An unclassified traced operator marks the count incomplete. Equal parameter layouts can have different checkpoint file sizes due to serialization.
- **Latency** measures CPU FP32 on one PyTorch inference thread using the bundled RGB image. First-mask timing includes the complete segmentation call for the first box. Cached timing starts after the image embedding exists and returns the second box prediction. Use the median of three timed calls after one warmup per model per path. The `--warmup` and `--samples` options can reduce these counts. Run on an otherwise idle machine. The report includes raw milliseconds, median, p95, and candidate-minus-original deltas.

## INT8 neck experiment

With the research environment active and COCO val2017 available, run:

```sh
python -m research.sam.int8_neck --coco-root /tmp/tinysegment-coco
```

This CPU experiment uses PyTorch's pinned eager QNNPACK backend. It folds BatchNorm
and quantizes the eight convolution filter banks in the four middle neck blocks
with symmetric per-output-channel INT8 weights. Per-tensor activation ranges
come from 16 eligible images outside the quality manifest. The command downloads
missing calibration JPEGs into the COCO val2017 directory. GELU and residual
additions run in FP32.

The command prints a new run directory. Its research-only checkpoint contains
the complete mixed-precision model. After reloading it, the experiment writes
COCO accuracy and fidelity, actual serialized sizes, a paired neck timing,
one paired full-path timing, and one FP32 no-op control to `report.json`.
Latency uses one CPU thread, one warmup, and three samples per model per path.
Checkpoint loading and weight packing are outside the timing window.

See the [INT8 neck experiment report](../../docs/sam-int8-neck.md) for the paired baseline results, operator profile, and decision.
