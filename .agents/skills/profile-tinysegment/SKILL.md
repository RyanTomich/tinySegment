---
name: profile-tinysegment
description: Profile TinySegment CPU latency from JPEG input to the first full-resolution mask, compare original and research SAM, or break a SAM block down by operator. Use for latency measurements and tables from existing benchmark reports.
---

# Profile TinySegment

Work from the repository root with its editable Python 3.12 environment. Use the weight setup in [AGENTS.md](../../../AGENTS.md). If asked to summarize an existing run, go directly to **Read the report**. For a fresh run, select the timing boundary first.

## Resident JPEG to first mask

Use the ExecuTorch environment and build the stage 3 and 4 programs for a paired comparison. Load the text encoder and detector once. Load each SAM with `tinysegment.pipeline.load_stage("segmenter", variant)`, using `baseline` and `int8_stage3_stage4`. [INT8 SAM build recipe](../../../models/efficientvit_sam_l0/int8_stage3_stage4/README.md)

- **Before the clock:** load and verify all checkpoints, load the compiled block, initialize runtime pools, and warm both models. Use five warmups per model and 30 requests per mode to match the existing reports. Keep the fixture, prompt, CPU settings and backend fixed across a model comparison. Set both PyTorch and ExecuTorch to one thread for comparison with the existing reports.
- **Inside the clock:** use monotonic wall time (`perf_counter_ns`). Start immediately before opening `tests/fixtures/bus.jpg`. Decode and orient it, encode `person`, run detection, call SAM’s `set_image` on the new image, and predict the first detected box. Stop when `predict` returns the full-resolution mask. This repeats image encoding on every request. The separate cached-prompt mode reuses only the CLIP embedding.
- **After the clock:** compare boxes, mask dimensions and predictions, compute hashes or IoU, and write results. Later boxes, overlay rendering and output serialization fall outside this boundary. In a custom comparison, alternate which model runs first and perform detailed instrumentation in a separate pass.

## A single block and its operators

For the six-row stage 4 table, reuse a successful stage 4 experiment report if available. For a fresh FP32 measurement, load baseline SAM and read `tests/fixtures/bus.jpg` with `tinysegment.pipeline.read_image`. The variant builder exposes `entry(sam)`, `capture(sam, rgb)` and `operations(block)`. Capture the entry input and expected output under `torch.inference_mode()` and verify exact replay with `torch.testing.assert_close(block(x), expected, atol=0, rtol=0)`. Use three warmups and five timed calls to match the original FP32 block reports. [Builder](../../../models/efficientvit_sam_l0/int8_stage3_stage4/build.py)

For another block, capture its real input and output from the image pass, preserve dtype and strides, and verify replay before benchmarking. Time the intact block with hooks removed. In a separate pass, bracket the leaf operators with pre/post hooks and take the median for each operator. Treat parent and child timers as inclusive. For a compiled block, inspect its native runtime trace to identify fused operations, conversions, layout copies and fallbacks. Charge all boundary conversions to the whole-block timing. A kernel-only timing starts after its input has the required dtype and layout.

## Read the report

Set `REPORT` to the path of an existing resident or stage 4 `report.json`. Generate either requested Markdown table from measured values:

```sh
python .agents/skills/profile-tinysegment/scripts/report_table.py "$REPORT"
```

For resident latency, quote the median and p95 from the **uninstrumented** `encode_prompt.total` under `summary_ms`. Use the model keys in `control_model` and `candidate_model` for paired reports. Older reports without these fields use `original` and `stage4_int8`. For time saved, use `paired_savings.encode_prompt`; positive values mean faster research SAM. Report first-mask IoU when comparing models. Label cached-prompt results separately.

The component table uses `profiled_encode_prompt`. Its overhead row is the median of the per-request remainder after JPEG, text, detection, image encoder, and first-box prediction. Compare the profiled and uninstrumented totals to check timer overhead. Component medians need not sum to the request median. The operator table reads `original_profile_median_ms`; use the separate intact timings for whole-block latency. Link the source report and identify the CPU, thread count, fixture and sample count with the result.
