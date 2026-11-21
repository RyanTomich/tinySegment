"""Export and time backbone entry blocks on CPU. Stage 4 also tests its winner end to end."""
import argparse
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter_ns

import numpy as np
from PIL import Image
import torch
from tinysegment.experimental_models.efficientvit_sam.int8_stage3_stage4.build import (
    capture, entry, folded_block, operations,
)

from tinysegment import load_detector, load_segmenter, load_text_encoder
from tinysegment.pipeline import read_image
from tinysegment.weights import sha256
from .checkpoints import allocate_run
from .evaluate import DEFAULT_IMAGE
from .int8_neck import calibration_paths
from .quality import mask_iou
from .model import RuntimeBlock


def samples(calls, rounds):
    names = list(calls)
    for _ in range(3):
        for call in calls.values():
            call()
    result = {name: [] for name in names}
    for i in range(rounds):
        offset = i % len(names)
        for name in names[offset:] + names[:offset]:
            start = perf_counter_ns()
            calls[name]()
            result[name].append((perf_counter_ns() - start) / 1e6)
    return result


def saving(control, candidate):
    differences = np.subtract(control, candidate)
    rng = np.random.default_rng(0)
    estimates = np.median(rng.choice(differences, size=(10000, len(differences))), axis=1)
    return {'median_ms': float(np.median(differences)),
            'ci95_ms': np.percentile(estimates, [2.5, 97.5]).tolist()}


def profile_original(block, x):
    timings = {}
    starts = {}
    prepared = {}
    def before(name, module, args):
        if name in ('expansion', 'depthwise', 'projection'):
            prepared[name] = args[0]
        starts[name] = perf_counter_ns()

    def after(name, module, args, output):
        timings.setdefault(name, []).append((perf_counter_ns() - starts[name]) / 1e6)

    ops = operations(block)
    with ExitStack() as stack:
        for name, module in ops.items():
            stack.enter_context(module.register_forward_pre_hook(partial(before, name)))
            stack.enter_context(module.register_forward_hook(partial(after, name)))
        for _ in range(5):
            block(x)
    kernels = {name: partial(ops[name], value) for name, value in prepared.items()}
    return timings, samples(kernels, 5)


def errors(output, reference):
    if output.shape != reference.shape or not torch.isfinite(output).all():
        raise RuntimeError('Invalid candidate output')
    diff = (output - reference).float()
    return {'shape': list(output.shape), 'dtype': str(output.dtype), 'finite': True,
            'max_abs': diff.abs().max().item(), 'rmse': diff.square().mean().sqrt().item()}


def configure_threads():
    from executorch.extension.pybindings import portable_lib
    torch.set_num_threads(1)
    portable_lib._unsafe_reset_threadpool(1)
    if portable_lib._threadpool_get_thread_count() != 1:
        raise RuntimeError('Expected one ExecuTorch worker')


@torch.inference_mode()
def first_mask(program, destination):
    configure_threads()
    text, detector = load_text_encoder(), load_detector()
    sam = load_segmenter()
    if program:
        sam.model.image_encoder.backbone.stages[4].op_list[0] = RuntimeBlock(program).eval()
    prompts = ['person']
    # Exercise lazy kernels on a different image. The measured call recomputes every feature.
    blank = Image.new('RGB', (810, 1080))
    detector.detect(blank, text.encode(prompts), prompts)
    sam.segment(blank, [[50, 398, 247, 904]])
    sam.predictor.reset_image()
    start = perf_counter_ns()
    rgb = read_image(DEFAULT_IMAGE)
    boxes = detector.detect(rgb, text.encode(prompts), prompts)['boxes']
    sam.predictor.set_image(np.asarray(rgb.convert('RGB')))
    masks, scores, _ = sam.predictor.predict(box=np.asarray(boxes[0], dtype=np.float32),
                                            multimask_output=False)
    elapsed = (perf_counter_ns() - start) / 1e6
    with destination.open('xb') as stream:
        np.savez(stream, ms=elapsed, mask=masks[0], score=scores[0], box=boxes[0])


def paired_images(program):
    pairs = []
    with tempfile.TemporaryDirectory() as temp:
        for i in range(5):
            order = ['baseline', 'candidate']
            if i % 2:
                order.reverse()
            results = {}
            for name in order:
                path = Path(temp) / f'{i}-{name}.npz'
                cmd = [sys.executable, '-m', __spec__.name, '--first-mask-output', str(path)]
                if name == 'candidate':
                    cmd.extend(['--program', str(program)])
                subprocess.run(cmd, check=True)
                with np.load(path) as data:
                    results[name] = dict(data)
            a, b = results['baseline'], results['candidate']
            np.testing.assert_array_equal(a['box'], b['box'])
            left, right = a['mask'], b['mask']
            pairs.append({'order': order, 'baseline_ms': float(a['ms']), 'candidate_ms': float(b['ms']),
                          'box': a['box'].tolist(), 'scores': [float(a['score']), float(b['score'])],
                          'pixel_agreement': float(np.mean(left == right)),
                          'mask_iou': mask_iou(left, right)})
    baseline = [p['baseline_ms'] for p in pairs]
    candidate = [p['candidate_ms'] for p in pairs]
    return {'pairs': pairs, 'baseline_median_ms': float(np.median(baseline)),
            'candidate_median_ms': float(np.median(candidate)), 'saving': saving(baseline, candidate)}


@torch.no_grad()
def run(coco_root, directory, report, stage_index=4):
    from .stage4_xnnpack import export_block, trace
    configure_threads()
    sam = load_segmenter()
    block = entry(sam, stage_index)
    x, reference = capture(sam, read_image(DEFAULT_IMAGE), stage_index)
    expected_shape = {3: (1, 128, 64, 64), 4: (1, 256, 32, 32)}[stage_index]
    if x.shape != expected_shape or x.dtype != torch.float32:
        raise RuntimeError(f'Unexpected stage {stage_index} entry tensor')
    torch.testing.assert_close(block(x), reference, atol=0, rtol=0)
    eager = partial(block, x)
    report.update(checkpoint_sha256=sha256(sam.checkpoint), image_sha256=sha256(DEFAULT_IMAGE),
                  input_shape=list(x.shape), input_stride=list(x.stride()), replay_exact=True,
                  original_baseline_ms=samples({'eager': eager}, 5)['eager'])
    breakdown, kernels = profile_original(block, x)
    report.update(original_profile_ms=breakdown, eager_prepared_convolutions_ms=kernels)
    report['original_profile_median_ms'] = {k: float(np.median(v)) for k, v in breakdown.items()}
    paths = calibration_paths(coco_root)
    calibration = [capture(sam, read_image(path), stage_index)[0] for path in paths]
    report['calibration_images'] = [p.name for p in paths]
    calls = {'eager': eager}
    report['exports'] = {}
    formats = ('fp32', 'int8') if stage_index == 3 else ('fp32', 'int8', 'fp16')
    for name in formats:
        print(f'Exporting and tracing {name}', flush=True)
        try:
            path, graph = export_block(name, folded_block(block), x, calibration, directory)
            candidate = RuntimeBlock(path)
            output = candidate(x)
            diagnostic = trace(path, x)
        except (RuntimeError, NotImplementedError) as exc:
            if name != 'fp16':
                raise
            report['exports'][name] = {'status': 'unavailable', 'reason': str(exc)}
            continue
        stats = errors(output, reference)
        report['exports'][name] = {'sha256': sha256(path), 'output': stats, 'graph': graph, **diagnostic}
        if name == 'fp32':
            torch.testing.assert_close(output, reference, rtol=1e-3, atol=2e-4)
        native = diagnostic['native_conv_operators']
        if name == 'int8' and not all('Q' in op and '8' in op for op in native):
            raise RuntimeError(f'Could not verify three native quantized convolutions: {native}')
        if name == 'fp16' and not all('F16' in op for op in native):
            report['exports'][name]['status'] = 'native FP16 convolutions unverified'
            continue
        calls[name] = partial(candidate, x)
    raw = samples(calls, 20)
    report['whole_block_ms'] = raw
    report['median_ms'] = {k: float(np.median(v)) for k, v in raw.items()}
    report['savings'] = {k: {base: saving(raw[base], raw[k]) for base in ('eager', 'fp32')}
                         for k in raw if k not in ('eager', 'fp32')}
    winners = [k for k, comparisons in report['savings'].items()
               if all(s['ci95_ms'][0] > 0 for s in comparisons.values())]
    report['fp32_vs_eager'] = saving(raw['eager'], raw['fp32'])
    winner = min(winners, key=report['median_ms'].get) if winners else None
    if winner is None and report['fp32_vs_eager']['ci95_ms'][0] > 0:
        winner = 'fp32'
    report['winner'] = winner
    print(report['median_ms'], 'winner:', winner, flush=True)
    if winner and stage_index == 4:
        report['first_mask'] = paired_images(directory.path / f'{winner}.pte')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path)
    parser.add_argument('--stage', type=int, choices=(3, 4), default=4)
    parser.add_argument('--first-mask-output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--program', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.first_mask_output:
        first_mask(args.program, args.first_mask_output)
        return
    if args.coco_root is None:
        parser.error('--coco-root is required')
    directory = allocate_run(f'stage{args.stage}-entry')
    print(directory.path, flush=True)
    report = {'platform': platform.platform(), 'cpu': torch.cpu.get_capabilities().get('cpu_name'),
              'versions': {p: version(p) for p in ('torch', 'executorch', 'torchao')},
              'stage': args.stage, 'threads': 1, 'warmups': 3, 'initial_samples': 5, 'comparison_rounds': 20}
    try:
        run(args.coco_root, directory, report, args.stage)
        report['status'] = 'success'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        directory.json('report.json', report)


if __name__ == '__main__':
    main()
