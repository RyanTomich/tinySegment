# EfficientViT-SAM L0 experiment: INT8 encoder neck

## Initial state and hypothesis

TinySegment runs the FP32 image encoder once per image. Its neck includes four residual blocks, each with a dense 3×3 convolution followed by a 1×1 convolution. Together these eight convolutions account for about 31% of encoder multiply–accumulates. We hypothesized that quantizing them would reduce first-mask latency and checkpoint size while preserving box-prompted mask quality. [Baseline size and operations](../runs/sam/20261118T031615850280Z-first-baseline-39e47ebd/evaluations/20261118T034229144278Z-size-ops-99344181/report.json)

## Intervention

We folded BatchNorm into each convolution, quantized weights to signed INT8 with one scale per output channel, and calibrated activation ranges on 16 COCO images excluded from evaluation. QNNPACK executed the eight convolutions. GELU and residual additions ran in FP32, requiring activation conversions around each convolution. We saved and reloaded the complete candidate before comparing it with the original model. [Experiment implementation](../research/sam/int8_neck.py)

## Results against baseline

Latency was measured on the bundled bus image using one Apple M4 Pro CPU thread. The recorded benchmark used 10 warmups and 100 timed calls per model per path. First-mask and cached-box ranges span three paired repeats. Quality is mean per-instance mask IoU against COCO annotations using the same ground-truth boxes. [Paired benchmark results](../runs/sam/20261118T041932721956Z-int8-neck-637d7efa/report.json) · [COCO quality results](../runs/sam/20261118T041932721956Z-int8-neck-637d7efa/quality.json)

| Metric | FP32 baseline | INT8 neck |
|---|---:|---:|
| Complete checkpoint | 132.95 MiB | 125.49 MiB |
| Four neck blocks, median | 26.2 ms | 207.0 ms |
| First mask, median | 337–339 ms | 519–521 ms |
| First mask, p95 | 352–367 ms | 527–540 ms |
| Cached box, median | 22.4–22.5 ms | 23.0–23.1 ms |
| COCO mean IoU, 100 images and 742 instances | 78.411 | 78.374 |

### Operator profile

A separate, single warmed-up pass through the INT8 blocks took 210.2 ms. The eight convolution calls consumed 194.9 ms (92.7%). The 16 explicit activation conversions consumed 1.25 ms (0.6%).

| Operator group | Time |
|---|---:|
| Four 3×3 convolutions | 174.4 ms |
| Four 1×1 convolutions | 20.5 ms |
| Four GELU activations | 11.7 ms |
| Four residual additions | 1.6 ms |
| Eight quantizations and eight dequantizations | 1.25 ms |
| Other overhead | 0.8 ms |

## Decision

Keep the FP32 neck for this CPU workload. INT8 saved 7.46 MiB (5.61%) and lowered mean IoU by 0.037 points, but first-mask latency increased about 54% in every repeat. Removing explicit conversions could recover only about 1.25 ms of the roughly 182 ms regression. The next latency experiment should compare one 3×3 layer in FP32 and QNNPACK INT8 using the same real neck input, BatchNorm folded in both versions, and input quantization outside timing. Use one warmup and the median of three calls. If INT8 remains slower, use that layer to screen alternative backends before expanding coverage. For a size-only experiment, test storing INT8 weights and expanding them to FP32 once at load.
