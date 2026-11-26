# Baseline validation

Validated on macOS arm64, Python 3.12, CPU FP32 with the versions pinned in `requirements.lock`.

## Real-model comparison

Input: bundled `tests/fixtures/bus.jpg` (810 × 1080).
Prompts: `person`, `bus`. Reference: the original locally installed model implementations and checkpoints recorded by the provenance and weight manifests.

| Output | Observed comparison |
|---|---|
| CLIP normalized embeddings | Exact match (maximum absolute error 0) |
| YOLO detections | Same five detections, ordering, and class IDs |
| YOLO box coordinates | Maximum absolute error 0.00030517578125 pixels |
| YOLO confidence scores | Maximum absolute error 0.0000037550926208496094 |
| SAM clean-image masks | All five masks exactly equal; every mask IoU 1.0 |
| Legacy combined overlay, prompt `person` | Pixel-for-pixel identical to the original demo PNG |

The reference high-level YOLO predictor fuses layers; TinySegment retains unfused modules for research. Comparison acceptance tolerances were 1e-6 for embeddings, 1e-3 pixels for boxes, 1e-5 for detection confidence, matching class IDs, and mask IoU ≥ 0.999. This is a smoke-test fixture, not a dataset-level accuracy or performance benchmark.

## Automated checks

Eight unittest tests passed, including two real-weight integration tests covering:

- Independent CLIP, detector, and segmenter commands, plus clean and legacy pipelines.
- Commands launched from a temporary working directory with Python audit hooks rejecting network connections, DNS queries, and file access to sibling projects.
- Model imports resolving to this checkout's vendored packages.
- No CLIP/SAM imports during detector execution; no CLIP/YOLO imports during segmenter execution.
- Normalized finite embeddings, multiple detections, correctly shaped masks, and no-detection output.
- Empty prompts, invalid images/boxes, missing weights, incompatible checkpoints, EXIF orientation, and input-overwrite protection.
- Editable module replacement and state-dictionary save/load round trips for all three models.

Checkpoint SHA-256 checks passed for all copied weights and for fresh downloads of all three checkpoints into a temporary directory. `pip check` found no broken dependencies. Wheel construction succeeded; package inspection confirmed tokenizer assets, YOLO configuration, checkpoint manifest, and all four model-source licenses are included. The non-editable wheel was also installed into a temporary directory and successfully ran the complete three-model pipeline with all model imports resolving to that installation.

CUDA, MPS, training, quantization, exported runtimes, and cross-platform reproducibility have not been validated.

## Historical SAM snapshot validation

Before the SAM evaluation tools were removed, the combined suite passed 16 tests, including real-weight tests for creating a baseline run, starting another run from its checkpoint, and using a candidate in the complete pipeline. Safety tests covered original-storage rejection, output aliases (symlinks and hard links), unique run allocation, collisions, checkpoint no-clobber behavior, read-only setup, checksum mismatches, failed serialization, incompatible model state, and interrupted experiments.

A saved baseline run was also exercised through both `tinysegment segment --checkpoint` and `tinysegment run --segmenter-checkpoint`. Its two fixed-box masks exactly matched the originals (mean mask IoU 1.0), and its 34,792,784 parameters survived save/reload. All three original checkpoint hashes matched their recorded values after the test.
