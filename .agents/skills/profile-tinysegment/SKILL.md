---
name: profile-tinysegment
description: Profile TinySegment CPU latency from JPEG input to the first full-resolution mask, compare original and research SAM, or break a SAM block down by operator. Use for latency measurements and tables from existing benchmark reports.
---

# Profile TinySegment

Work from the repository root with its editable Python 3.12 environment. Use the weight setup in [AGENTS.md](../../../AGENTS.md) and the [SAM research guide](../../../research/sam/README.md). If asked to summarize an existing run, go directly to **Read the report**. For a fresh run, select the timing boundary first.

## Resident JPEG to first mask

Use the ExecuTorch environment with the two installed INT8 entry programs described in the [SAM research guide](../../../research/sam/README.md). The default comparison runs original FP32 SAM and research SAM with stages 3 and 4 quantized.

```sh
OMP_NUM_THREADS=1 python -m research.sam.profile_resident
# Include the paired 100-image COCO evaluation after timing
OMP_NUM_THREADS=1 python -m research.sam.profile_resident --coco-root "$COCO_ROOT"
# Original SAM alone also works in the standard project environment
OMP_NUM_THREADS=1 python -m research.sam.profile_resident --original-only
```

- **Before the clock:** load and verify all checkpoints, load the compiled block, initialize runtime pools, and warm both models. The resident benchmark uses five warmups per model and 30 requests per mode. Keep the fixture, prompt, CPU settings and backend fixed across a model comparison. The stock profilers set PyTorch and the compared ExecuTorch runtime to one thread internally. To study parallelism, configure and verify both pools explicitly.
- **Inside the clock:** use monotonic wall time (`perf_counter_ns`). Start immediately before opening the local bus JPEG. Decode and orient it, encode `person`, run detection, call SAM’s `set_image` on the new image, and predict the first detected box. Stop when `predict` returns the full-resolution mask. This repeats image encoding on every request. The separate cached-prompt mode reuses only the CLIP embedding.
- **After the clock:** compare boxes, mask dimensions and predictions, compute hashes or IoU, and write results. Later boxes, overlay rendering and output serialization fall outside this boundary. In a custom comparison, alternate which model runs first and perform detailed instrumentation in a separate pass.

## A single block and its operators

For the six-row stage 4 table, reuse a successful stage 4 experiment report if available. To measure only the original FP32 block, the following runs in the standard project environment without calibration or a compiled candidate. It uses the same capture and timing routines as the full stage 4 experiment.

```sh
OMP_NUM_THREADS=1 python - <<'PY'
from functools import partial
from statistics import median
import torch
from tinysegment import load_segmenter
from tinysegment.pipeline import read_image
from research.sam.checkpoints import allocate_run
from research.sam.evaluate import DEFAULT_IMAGE
from research.sam.stage4 import capture, entry, profile_original, samples

torch.set_num_threads(1)
with torch.inference_mode():
    sam = load_segmenter()
    block = entry(sam)
    x, expected = capture(sam, read_image(DEFAULT_IMAGE))
    torch.testing.assert_close(block(x), expected, atol=0, rtol=0)
    intact = samples({'block': partial(block, x)}, 5)['block']
    operators, _ = profile_original(block, x)
report = {
    'original_baseline_ms': intact,
    'original_profile_ms': operators,
    'original_profile_median_ms': {k: median(v) for k, v in operators.items()},
    'input_shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype),
    'cpu': torch.cpu.get_capabilities().get('cpu_name'), 'torch': str(torch.__version__),
    'threads': torch.get_num_threads(), 'warmups': 3, 'timed_calls': 5,
}
print(allocate_run('stage4-fp32-operators').json('report.json', report))
PY
```

For another block, capture its real input and output from the image pass, preserve dtype and strides, and verify replay before benchmarking. Time the intact block with hooks removed. In a separate pass, bracket the leaf operators with pre/post hooks and take the median for each operator. Treat parent and child timers as inclusive. For a compiled block, inspect its native runtime trace to identify fused operations, conversions, layout copies and fallbacks. Charge all boundary conversions to the whole-block timing. A kernel-only timing starts after its input has the required dtype and layout.

## Read the report

Set `REPORT` to the specific `report.json` printed by the chosen benchmark. Generate either requested Markdown table from measured values:

```sh
python .agents/skills/profile-tinysegment/scripts/report_table.py "$REPORT"
```

For resident latency, quote the median and p95 from the **uninstrumented** `encode_prompt.total` under `summary_ms`. Use the model keys in `control_model` and `candidate_model` for paired reports. Older reports without these fields use `original` and `stage4_int8`. For time saved, use `paired_savings.encode_prompt`; positive values mean faster research SAM. Report first-mask IoU when comparing models. Label cached-prompt results separately.

The component table uses `profiled_encode_prompt`. Its overhead row is the median of the per-request remainder after JPEG, text, detection, image encoder, and first-box prediction. Compare the profiled and uninstrumented totals to check timer overhead. Component medians need not sum to the request median. The operator table reads `original_profile_median_ms`; use the separate intact timings for whole-block latency. Link the source report and identify the CPU, thread count, fixture and sample count with the result.
