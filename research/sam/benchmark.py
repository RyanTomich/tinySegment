"""Benchmark one aspect of a saved EfficientViT-SAM checkpoint."""
import argparse
from functools import partial
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
import torch

from tinysegment import load_segmenter
from tinysegment.weights import sha256
from .evaluate import DEFAULT_BOXES, DEFAULT_IMAGE, comparison, evaluation_report, read_fixture


WARMUP = 1
SAMPLES = 3


def paired_samples(calls, warmup=WARMUP, samples=SAMPLES, clock=perf_counter_ns):
    if not 0 <= warmup <= WARMUP or not 1 <= samples <= SAMPLES:
        raise ValueError(f"Expected 0–{WARMUP} warmups and 1–{SAMPLES} timed samples")
    for _ in range(warmup):
        for call in calls:
            call()
    values = [[], []]
    for pair in range(samples):
        for index in ((1, 0) if pair % 2 else (0, 1)):
            start = clock()
            calls[index]()
            values[index].append((clock() - start) / 1e6)
    return {'original_ms': values[0], 'candidate_ms': values[1],
            'median_ms': comparison(*(float(np.median(v)) for v in values)),
            'p95_ms': comparison(*(float(np.percentile(v, 95)) for v in values))}


def latency(stages, warmup=WARMUP, samples=SAMPLES):
    rgb, boxes = read_fixture(DEFAULT_IMAGE, DEFAULT_BOXES)
    first_calls = [partial(stage.segment, rgb, boxes[:1]) for stage in stages]
    first_mask = paired_samples(first_calls, warmup, samples)
    for stage in stages:
        stage.predictor.set_image(np.asarray(rgb))
    cached_calls = [partial(stage.predictor.predict, box=boxes[1], multimask_output=False) for stage in stages]
    return {'first_mask': first_mask,
            'cached_box': paired_samples(cached_calls, warmup, samples),
            'cpu': torch.cpu.get_capabilities().get('cpu_name') or platform.machine(),
            'os': platform.platform(),
            'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'image_sha256': sha256(DEFAULT_IMAGE), 'boxes_sha256': sha256(DEFAULT_BOXES)}


def benchmark(kind, run, checkpoint='checkpoint.pt', **settings):
    if kind == 'quality' and 'coco_root' in settings:
        settings['coco_root'] = str(Path(settings['coco_root']).resolve())
        settings.setdefault('full', False)
    if kind == 'latency':
        settings.setdefault('warmup', WARMUP)
        settings.setdefault('samples', SAMPLES)
    report = {'benchmark': kind, 'device': 'cpu', 'dtype': 'float32', 'settings': settings}
    with evaluation_report(run, checkpoint, kind, report) as (original, candidate, path):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.inference_mode():
                stages = [load_segmenter(checkpoint=p) for p in (original, candidate)]
                if kind == 'quality':
                    from .quality import quality
                    report['metrics'] = quality(*stages, settings['coco_root'], settings.get('full', False))
                    report['pycocotools'] = version('pycocotools')
                elif kind == 'size-ops':
                    from .size_ops import size_ops
                    report['metrics'] = size_ops(stages)
                    report['fvcore'] = version('fvcore')
                elif kind == 'latency':
                    report['metrics'] = latency(stages, settings['warmup'], settings['samples'])
                else:
                    raise ValueError(f'Unknown benchmark: {kind}')
        finally:
            torch.set_num_threads(previous_threads)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    for name in ('quality', 'size-ops', 'latency'):
        sub = commands.add_parser(name)
        sub.add_argument('run', type=Path)
        sub.add_argument('--checkpoint', default='checkpoint.pt')
        if name == 'quality':
            sub.add_argument('--coco-root', required=True)
            sub.add_argument('--full', action='store_true')
        if name == 'latency':
            sub.add_argument('--warmup', type=int, choices=range(WARMUP + 1), default=WARMUP)
            sub.add_argument('--samples', type=int, choices=range(1, SAMPLES + 1), default=SAMPLES)
    try:
        benchmark(**vars(parser.parse_args()))
    except (ImportError, ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
