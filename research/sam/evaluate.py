"""Compare a saved SAM checkpoint with the original on a shared fixture."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from tinysegment import load_segmenter
from tinysegment.pipeline import read_image
from tinysegment.weights import sha256, verify_baseline
from .checkpoints import PROJECT_ROOT, RUNS_ROOT, allocate_run, reject_symlinks

DEFAULT_IMAGE = PROJECT_ROOT / 'vendor/ultralytics/assets/bus.jpg'
DEFAULT_BOXES = Path(__file__).with_name('fixture.json')


def comparison(original, candidate):
    delta = candidate - original
    return {'original': original, 'candidate': candidate, 'delta': delta,
            'delta_pct': 100 * delta / original if original else None}


def compare_masks(reference, candidate):
    left, right = reference['masks'], candidate['masks']
    if left.shape != right.shape or not np.isfinite(candidate['scores']).all():
        raise ValueError('Candidate returned invalid mask shapes or nonfinite scores')
    intersection = (left & right).sum(axis=(1, 2))
    union = (left | right).sum(axis=(1, 2))
    ious = np.divide(intersection, union, out=np.ones(len(union), dtype=float), where=union != 0)
    return {'pixel_agreement': float(np.mean(left == right)), 'mask_iou': ious.tolist(), 'mean_mask_iou': float(ious.mean())}


@contextmanager
def evaluation_report(run, checkpoint, name, report):
    run = Path(run).absolute()
    reject_symlinks(run)
    if not run.is_dir() or run.resolve().parent != RUNS_ROOT.resolve():
        raise ValueError(f'Expected an existing run directly inside {RUNS_ROOT}')
    if not checkpoint or Path(checkpoint).name != checkpoint or checkpoint in {'.', '..'}:
        raise ValueError('Checkpoint must be a filename inside the run')
    candidate = run / checkpoint
    reject_symlinks(candidate)
    directory = allocate_run(name, run / 'evaluations')
    report['checkpoint'] = checkpoint
    try:
        config = json.loads((run / 'config.json').read_text())
        original = Path(config['original_checkpoint'])
        report['original_checkpoint'] = str(original)
        report['original_sha256'] = verify_baseline('segmenter', original)
        report['output_sha256'] = sha256(candidate)
        yield original, candidate, directory.path / 'report.json'
        report['status'] = 'success'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        path = directory.json('report.json', report)
        print(f'Evaluation report: {path}', flush=True)


def evaluate(run, *, checkpoint='checkpoint.pt', image=DEFAULT_IMAGE, boxes=DEFAULT_BOXES):
    report = {'image': str(Path(image).resolve()), 'boxes_file': str(Path(boxes).resolve())}
    with evaluation_report(run, checkpoint, 'eval', report) as (original, candidate, path):
        rgb, prompts = read_fixture(image, boxes)
        report.update(image_sha256=sha256(image), boxes_sha256=sha256(boxes),
                      image_size=list(rgb.size), boxes=prompts.tolist())

        baseline = load_segmenter(checkpoint=original)
        reference = baseline.segment(rgb, prompts)
        report['original_parameters'] = sum(p.numel() for p in baseline.model.parameters())
        del baseline
        model = load_segmenter(checkpoint=candidate)
        report.update(baseline_fidelity=compare_masks(reference, model.segment(rgb, prompts)),
                      parameters=sum(p.numel() for p in model.model.parameters()),
                      checkpoint_bytes=candidate.stat().st_size,
                      original_checkpoint_bytes=original.stat().st_size,
                      pipeline_compatible=True)
    return path


def read_fixture(image, boxes):
    image_path, boxes_path = Path(image).resolve(), Path(boxes).resolve()
    rgb = read_image(image_path)
    fixture = json.loads(boxes_path.read_text())
    if fixture.get('image_size', list(rgb.size)) != list(rgb.size):
        raise ValueError('Boxes image_size does not match the EXIF-corrected image')
    prompts = np.asarray(fixture['boxes'], dtype=np.float32)
    if prompts.ndim != 2 or prompts.shape[1] != 4 or len(prompts) == 0 or not np.isfinite(prompts).all() or np.any(prompts[:, 2:] <= prompts[:, :2]):
        raise ValueError('Validation requires nonempty, finite [N, 4] xyxy boxes with positive areas')
    return rgb, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--boxes', type=Path)
    args = parser.parse_args()
    if (args.image is None) != (args.boxes is None):
        parser.error('Supply --image and --boxes together, or use the default fixture')
    try:
        evaluate(args.run, checkpoint=args.checkpoint, image=args.image or DEFAULT_IMAGE,
                 boxes=args.boxes or DEFAULT_BOXES)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
