"""Apply a SAM recipe and save its checkpoint in a new run."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from tinysegment import load_segmenter
from tinysegment.models import Segmenter
from tinysegment.weights import checkpoint_path, sha256, verify_baseline
from .checkpoints import PROJECT_ROOT, allocate_run
from .inspect import describe
from .recipes import RECIPES


def code_identity():
    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(PROJECT_ROOT), *args], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    paths = []
    for relative in ['research/sam', 'src/tinysegment', 'vendor/efficientvit', 'vendor/segment_anything']:
        paths.extend((PROJECT_ROOT / relative).rglob('*.py'))
    paths.append(PROJECT_ROOT / 'src/tinysegment/weights.json')
    status = git('status', '--porcelain')
    return {
        'git_revision': git('rev-parse', 'HEAD'),
        'git_dirty': bool(status) if status is not None else None,
        'git_status': status,
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted(paths)},
    }


def execute(*, name='baseline', recipe='baseline', input_checkpoint=None,
            weights_dir=None, seed=0):
    if recipe not in RECIPES:
        raise ValueError(f'Unknown recipe: {recipe}')
    # Verify the true original even when starting from a previous candidate.
    original = checkpoint_path('segmenter', weights_dir=weights_dir)
    input_path = checkpoint_path('segmenter', weights_dir=weights_dir, checkpoint=input_checkpoint)
    config = {
        'name': name, 'recipe': recipe, 'model': 'efficientvit-sam-l0',
        'device': 'cpu', 'dtype': 'float32', 'seed': seed,
        'input_checkpoint': str(input_path), 'original_checkpoint': str(original),
    }
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model': config['model'], 'format': 'pytorch-state-dict',
        'input_sha256': sha256(input_path), 'original_sha256': sha256(original),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {package: version(package) for package in ['torch', 'torchvision', 'numpy', 'pillow']},
        'code': code_identity(),
    }
    directory = allocate_run(name)
    print(f'Run directory: {directory.path}', flush=True)
    directory.json('config.json', config)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        stage = load_segmenter(checkpoint=input_path, weights_dir=weights_dir)
        model = RECIPES[recipe](stage.model, dict(config))
        if not isinstance(model, torch.nn.Module):
            raise ValueError('A recipe must return a PyTorch model')
        if any(value.is_quantized or (value.is_floating_point() and value.dtype != torch.float32)
               for value in model.state_dict().values() if isinstance(value, torch.Tensor)):
            raise ValueError('This workflow currently supports eager FP32 L0 checkpoints only')
        candidate = Segmenter(model, input_path)
        directory.text('inspection.txt', 'Candidate before saving; checkpoint below identifies its input.\n\n' + describe(candidate))
        saved = directory.checkpoint(candidate.model)
        # Guard against accidental recipe-side changes to baseline storage.
        verify_baseline('segmenter', original)
        if sha256(input_path) != metadata['input_sha256']:
            raise ValueError('Experiment modified its input checkpoint')
        metadata.update(status='success', output_sha256=sha256(saved))
        directory.json('metadata.json', metadata)
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        metadata.update(status=status, error=f'{type(exc).__name__}: {exc}')
        try:
            verify_baseline('segmenter', original)
            metadata['original_unchanged'] = True
        except (OSError, ValueError):
            metadata['original_unchanged'] = False
        try:
            if not (directory.path / 'metadata.json').exists():
                directory.json('metadata.json', metadata)
        except (OSError, ValueError):
            pass  # An interrupted write must never turn into successful publication.
        raise
    print(f'Generated checkpoint: {saved}', flush=True)
    return directory.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--recipe', choices=sorted(RECIPES), default='baseline')
    parser.add_argument('--input-checkpoint', type=Path)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    try:
        execute(**vars(args))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
