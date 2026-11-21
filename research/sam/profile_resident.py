"""Compare resident SAM variants from opening a JPEG to returning the first full-resolution mask."""
import argparse
from contextlib import ExitStack
from functools import wraps
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
import torch

from tinysegment import pipeline
from tinysegment.weights import MANIFEST, sha256 as file_sha256
from .checkpoints import allocate_run
from .evaluate import DEFAULT_IMAGE


WARMUPS = 5
SAMPLES = 30


def instrument(stack, times, obj, method, name):
    original = getattr(obj, method)

    @wraps(original)
    def measured(*args, **kwargs):
        start = perf_counter_ns()
        result = original(*args, **kwargs)
        times[name] = times.get(name, 0) + (perf_counter_ns() - start) / 1e6
        return result

    stack.enter_context(patch.object(obj, method, measured))


def trace_request(stack, times, text, detector, sam):
    encoder = sam.model.image_encoder
    for obj, method, name in [
        (pipeline, 'read_image', 'jpeg'),
        (text, 'encode', 'text'),
        (detector, 'detect', 'detection'),
        (detector, 'preprocess', 'detector_preprocess'),
        (detector, 'forward', 'detector_forward'),
        (detector, 'postprocess', 'detector_postprocess'),
        (sam.predictor, 'set_image', 'set_image'),
        (encoder, 'forward', 'image_encoder'),
        (encoder.neck, 'forward', 'neck'),
        (encoder.neck.middle, 'forward', 'neck_middle'),
        (encoder.norm, 'forward', 'encoder_norm'),
        (sam.predictor, 'predict', 'predict_box'),
        (sam.model.prompt_encoder, 'forward', 'prompt_encoder'),
        (sam.model.prompt_encoder, 'get_dense_pe', 'position'),
        (sam.model.mask_decoder, 'forward', 'mask_decoder'),
        (sam.model, 'postprocess_masks', 'resize_mask'),
    ]:
        instrument(stack, times, obj, method, name)
    trace_backbone(stack, times, encoder.backbone)


def trace_backbone(stack, times, backbone):
    for i, stage in enumerate(backbone.stages):
        instrument(stack, times, stage, 'forward', f'stage_{i}')
        for j, block in enumerate(stage.op_list):
            instrument(stack, times, block, 'forward', f'stage_{i}_block_{j}')
    eager_main = getattr(backbone.stages[3].op_list[0], 'main', None)
    if eager_main is not None:
        for name, operator in eager_main.named_modules():
            if not list(operator.children()):
                instrument(stack, times, operator, 'forward', f'stage_3_entry_{name}')
    stage4 = backbone.stages[4].op_list
    instrument(stack, times, stage4[0], 'forward', 'stage_4_downsample')
    for block in stage4[1:]:
        instrument(stack, times, block.context_module, 'forward', 'stage_4_attention')
        instrument(stack, times, block.local_module, 'forward', 'stage_4_local')


def request(text, detector, sam, prompts, cached=None):
    start = perf_counter_ns()
    image = pipeline.read_image(DEFAULT_IMAGE)
    embeddings = text.encode(prompts) if cached is None else cached
    boxes = detector.detect(image, embeddings, prompts)['boxes']
    box = np.asarray(boxes[0], dtype=np.float32)
    sam.predictor.set_image(np.asarray(image.convert('RGB')))
    masks, scores, _ = sam.predictor.predict(box=box, multimask_output=False)
    elapsed = (perf_counter_ns() - start) / 1e6
    mask = masks[0]
    prediction = {
        'boxes': boxes, 'score': float(scores[0]), 'mask_pixels': int(mask.sum()),
        'mask_shape': list(mask.shape), 'mask_sha256': sha256(mask.tobytes()).hexdigest(),
    }
    return elapsed, prediction, mask


def summarize(samples):
    summary = {}
    for mode, rows in samples.items():
        summary[mode] = {}
        for key in rows[0]:
            median, p95 = np.percentile([r[key] for r in rows], [50, 95]).tolist()
            summary[mode][key] = {'median': median, 'p95': p95}
    return summary


def benchmark(coco_root=None, *, original_only=False):
    if original_only and coco_root is not None:
        raise ValueError('COCO comparison requires both original and research SAM')
    torch.set_num_threads(1)
    text, detector = pipeline.load_text_encoder(), pipeline.load_detector()
    models = {'original': pipeline.load_segmenter()}
    if not original_only:
        from .model import load_research_segmenter
        from .stage4 import configure_threads
        configure_threads()
        models['stage3_stage4_int8'] = load_research_segmenter()
    prompts = ['person']
    cached = text.encode(prompts)
    predictions, masks = {}, {}
    for name, sam in models.items():
        for i in range(WARMUPS):
            _, predictions[name], masks[name] = request(text, detector, sam, prompts, cached if i % 2 else None)

    modes = ['encode_prompt', 'cached_prompt', 'profiled_encode_prompt']
    samples = {name: {mode: [] for mode in modes} for name in models}
    names = list(models)
    for i in range(SAMPLES):
        offset = i % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            # Each pair uses the same prompt mode. Alternate which SAM goes first.
            for name in (names if i % 2 == 0 else names[::-1]):
                sam = models[name]
                times = {}
                with ExitStack() as stack:
                    if mode == 'profiled_encode_prompt':
                        trace_request(stack, times, text, detector, sam)
                    times['total'], prediction, _ = request(
                        text, detector, sam, prompts, cached if mode == 'cached_prompt' else None)
                if prediction != predictions[name]:
                    raise RuntimeError(f'{name} prediction changed in {mode} sample {i + 1}')
                if mode == 'profiled_encode_prompt':
                    times['other_request'] = times['total'] - sum(
                        times[k] for k in ('jpeg', 'text', 'detection', 'image_encoder', 'predict_box'))
                samples[name][mode].append(times)

    summaries = {name: summarize(rows) for name, rows in samples.items()}
    report = {
        'method': 'Resident CPU models. Local JPEG open through first predict return. Every request '
                  'reruns detection and SAM set_image. Only cached_prompt reuses CLIP output. '
                  'Output validation and timer installation are outside the clock. Component timers '
                  'are inclusive. Block 0 is the stage entry. Stage 3 entry operators are nested '
                  'inside block 0. Stage 4 downsample and stage_4_block_0 time the same block. '
                  'Stage 4 attention and local sum the four repeated blocks.',
        'warmups': WARMUPS, 'samples_per_mode': SAMPLES,
        'order': 'rotate prompt modes each round; alternate baseline/candidate within each pair',
        'platform': platform.platform(), 'python': platform.python_version(), 'torch': str(torch.__version__),
        'cpu': torch.cpu.get_capabilities().get('cpu_name', platform.machine()),
        'torch_intraop_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'image': str(DEFAULT_IMAGE), 'image_sha256': file_sha256(DEFAULT_IMAGE), 'prompts': prompts,
        'checkpoints': {name: spec['sha256'] for name, spec in MANIFEST.items()},
    }
    if original_only:
        report.update(prediction=predictions['original'], summary_ms=summaries['original'],
                      samples_ms=samples['original'])
        return allocate_run('fp32-resident-first-mask').json('report.json', report)

    from .quality import mask_iou, quality
    from .stage4 import saving
    control_name, candidate_name = names
    original, candidate = masks[control_name], masks[candidate_name]
    if predictions[control_name]['boxes'] != predictions[candidate_name]['boxes'] or original.shape != candidate.shape:
        raise RuntimeError('Comparators produced different boxes or mask dimensions')
    savings = {}
    for mode in modes:
        baseline = [r['total'] for r in samples[control_name][mode]]
        treatment = [r['total'] for r in samples[candidate_name][mode]]
        paired = saving(baseline, treatment)
        paired['median_paired_percent'] = float(np.median(100 * np.subtract(baseline, treatment) / baseline))
        savings[mode] = paired
    report.update(control_model=control_name, candidate_model=candidate_name,
                  treatment={key: models[candidate_name].metadata[key] for key in ('stage3_program', 'stage4_program')},
                  executorch=version('executorch'), executorch_threads=1, predictions=predictions,
                  first_mask_iou=mask_iou(original, candidate), pixel_agreement=float(np.mean(original == candidate)),
                  summary_ms=summaries, samples_ms=samples, paired_savings=savings)
    directory = allocate_run('original-vs-research-resident-first-mask')
    path = directory.json('report.json', report)
    if coco_root is not None:
        print(f'Resident report: {path}. Evaluating COCO quality.', flush=True)
        metrics = quality(models[control_name], models[candidate_name], coco_root)
        directory.json('quality.json', {'control_model': control_name, 'candidate_model': candidate_name, **metrics})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-only', action='store_true', help='Profile the FP32 control alone')
    parser.add_argument('--coco-root', type=Path, help='Also measure paired quality on the 100-image COCO manifest')
    args = parser.parse_args()
    if args.original_only and args.coco_root is not None:
        parser.error('--coco-root requires the paired benchmark')
    print(benchmark(args.coco_root, original_only=args.original_only), flush=True)


if __name__ == '__main__':
    main()
