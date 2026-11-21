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

## Stage 4 entry-block latency

`research.sam.stage4` benchmarks the downsampling entry block of backbone stage 4 by default. Pass `--stage 3` to export the stage 3 entry.
Use a separate Python 3.12 environment. On the validated macOS arm64 setup,
install the repository lockfile and editable checkout there, then install
`executorch==1.6.0 torchao==0.18.0`. Run from the checkout:

```sh
OMP_NUM_THREADS=1 python -m research.sam.stage4 --coco-root /tmp/tinysegment-coco
```

The benchmark verifies exact replay on the bus image, profiles the six original
operators, and calibrates static INT8 on the neck experiment's 16 held-out
COCO images. It compares eager FP32 with XNNPACK FP32, INT8 and, when verified,
FP16 convolutions. All block timings accept and return the original FP32 tensor
format. Three warmups precede five initial baseline timings and 20 interleaved
comparison rounds on one thread. The diagnostic trace is separate. For XNNPACK,
the individual convolution times are native operator events inside the real
block, after the runtime has prepared each input. The report lists layout and
dtype conversions separately and retains opaque runtime event names verbatim.

A reduced-precision candidate advances if the 95% bootstrap intervals for the
paired median time savings against both FP32 controls are positive. If only
XNNPACK FP32 beats eager, test that backend change. The final comparison runs
five pairs in separate processes with alternating order. Each process loads
all models and primes the runtime on a blank image before timing JPEG input,
text encoding, detection and the first full-resolution mask. The report contains
raw timings, first-mask IoU and pixel agreement, and paths to the run's compiled
programs are printed via the run directory. Results apply to the measured CPU
and runtime versions.

## Standard research SAM

Use `from research.sam.model import load_research_segmenter` and call
`load_research_segmenter()` in the ExecuTorch environment. It loads EfficientViT-SAM
L0 and replaces the downsampling entry blocks of backbone stages 3 and 4 with
ExecuTorch programs delegated to XNNPACK on CPU. Each block has a 1×1 expansion
convolution, a stride-2 depthwise convolution, and a 1×1 projection. Both accept
and return FP32 tensors, so the surrounding PyTorch encoder and SAM decoders
use the original tensor interface.

Both programs use static post-training INT8 quantization. During export, the
projection absorbs its BatchNorm parameters. TorchAO PT2E and the XNNPACK
quantizer assign symmetric INT8 weights per output channel and affine INT8
activations per tensor. Activation scales and zero points are calibrated using
real block inputs from 16 COCO images held out of the quality benchmark.
The native traces confirm all six convolutions use quantized XNNPACK kernels.

The two installed programs live in `runs/sam/research-model/`. For a fresh
checkout, run the stage 4 export above and repeat it with `--stage 3`.
Set `STAGE4_RUN` and `STAGE3_RUN` to the respective printed directories,
then install the files:

```sh
mkdir -p runs/sam/research-model
cp "$STAGE3_RUN/int8.pte" runs/sam/research-model/stage3-int8.pte
cp "$STAGE4_RUN/int8.pte" runs/sam/research-model/stage4-int8.pte
```

Compare the original FP32 model directly with the standard research model:

```sh
OMP_NUM_THREADS=1 python -m research.sam.profile_resident --coco-root /tmp/tinysegment-coco
```

The profiler loads both models, warms each five times, then alternates their
order over 30 pairs per mode. Each request opens the bus JPEG, runs detection,
recomputes SAM image features, and returns the first full-resolution mask.
The primary mode encodes `person` on every request. A second mode caches the
prompt embedding, and a third supplies component timings. The optional COCO
evaluation scores both models against identical ground-truth boxes on the
fixed 100-image manifest. The command writes raw timings and program hashes
to `report.json` and COCO metrics to `quality.json`. For a standalone FP32
latency measurement use `--original-only`.

On one Apple M4 Pro CPU thread, median JPEG-to-first-mask latency fell from
407.9 to 316.1 ms and p95 from 423.5 to 324.8 ms. The median paired saving
was 92.9 ms (22.9%), with a 95% bootstrap interval of 91.1–95.1 ms. On 742
COCO instances, mean box-prompted mask IoU against ground truth fell from
78.41 to 77.96, a loss of 0.45 points.
[Direct latency comparison](../../runs/sam/20261121T193243960535Z-original-vs-research-resident-first-mask-667c001c/report.json) ·
[COCO quality](../../runs/sam/20261121T193243960535Z-original-vs-research-resident-first-mask-667c001c/quality.json)

## Earlier stage-by-stage comparisons

On the single-thread M4 Pro bus-image benchmark, encoding `person` on each
request took 412.7 ms median and 424.6 ms p95 with original SAM. The stage 4
INT8 variant took 369.0 ms median and 377.4 ms p95. Across 30 pairs, the
median saving was 43.7 ms (10.6%). The stage 4 entry block went from 45.4 to
3.1 ms under the detailed timers. The first mask had 0.9953 IoU with the
original on this image. [Paired report](../../runs/sam/20261121T185729418799Z-stage4-int8-resident-first-mask-7a1b33f1/report.json)

The earlier profiler also timed each backbone block and the individual operators in
the stage 3 entry block. In a repeat on the same image and CPU, stage 3 was the
slowest encoder component in all 30 profiled requests. Its median was 88.9 ms,
followed by stage 4 at 65.3 ms. The stage 3 entry block took 55.9 ms, including
24.0 ms for its stride-2 depthwise convolution and 23.1 ms for the GELU after
the expansion convolution. These operator times are included in the entry block
and stage totals. [Block and operator timings](../../runs/sam/20261121T190430282244Z-stage4-int8-resident-first-mask-81e5da56/report.json)

### Adding stage 3 to the stage-4-only experimental baseline

On one M4 Pro CPU thread, the isolated stage 3 entry took 52.0 ms median
in eager FP32, 7.9 ms in XNNPACK FP32, and 3.3 ms in XNNPACK INT8.
With the prompt encoded on each request, median JPEG-to-first-mask latency
fell from 370.0 to 321.5 ms. The median within-pair saving was 46.0 ms
(12.5%, 95% bootstrap interval for milliseconds 44.5–48.1). On 742 COCO
instances, box-prompted mean IoU against ground truth fell from 78.43 to
77.96, a decrease of 0.47 points.
[Block timings and trace](../../runs/sam/20261121T192000566072Z-stage3-entry-12277050/report.json) ·
[Resident latency](../../runs/sam/20261121T192151897380Z-stage3-stage4-int8-resident-first-mask-2c471a9c/report.json) ·
[COCO quality](../../runs/sam/20261121T192151897380Z-stage3-stage4-int8-resident-first-mask-2c471a9c/quality.json)
