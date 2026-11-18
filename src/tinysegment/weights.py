"""Local checkpoints, baseline integrity, and protected artifact destinations."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

MANIFEST = json.loads(Path(__file__).with_name('weights.json').read_text())


def project_weights_dir():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'vendor').is_dir():
            return parent / 'weights'
    return Path.home() / '.cache' / 'tinysegment' / 'weights'


def default_weights_dir():
    return Path(os.environ.get('TINYSEGMENT_WEIGHTS_DIR', project_weights_dir())).expanduser()


def baseline_roots(weights_dir=None):
    roots = [project_weights_dir(), default_weights_dir()]
    if weights_dir is not None:
        roots.append(Path(weights_dir).expanduser())
    return {path.resolve() for path in roots}


def same_file(left, right):
    left, right = Path(left).expanduser(), Path(right).expanduser()
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_baseline(name, path):
    actual = sha256(path)
    if actual != MANIFEST[name]['sha256']:
        raise ValueError(f'Original {name} checkpoint checksum mismatch: {path}. Restore the original checkpoint with setup-weights; do not use modified weights as the baseline.')
    return actual


def checkpoint_path(name, weights_dir=None, checkpoint=None):
    path = Path(checkpoint).expanduser() if checkpoint else Path(weights_dir or default_weights_dir()).expanduser() / MANIFEST[name]['file']
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name} checkpoint: {path}. Run tinysegment setup-weights or provide --weights-dir/--checkpoint.')
    # Explicit aliases (including hard links) of originals remain protected baselines.
    if checkpoint is None or any(same_file(path, root / MANIFEST[name]['file']) for root in baseline_roots(weights_dir)):
        verify_baseline(name, path)
    return path.resolve()


def validate_output_path(path, *, weights_dir=None, inputs=()):
    """Reject original storage, checkpoint filenames, and aliases of input files."""
    path = Path(path).expanduser()
    resolved = path.resolve()
    if any(resolved == root or root in resolved.parents for root in baseline_roots(weights_dir)):
        raise ValueError(f'Output cannot be written inside original weights storage: {path}')
    if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'} or resolved.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'}:
        raise ValueError(f'Prediction output cannot overwrite a model checkpoint: {path}')
    protected = [value for value in inputs if value is not None]
    protected.extend(root / entry['file'] for root in baseline_roots(weights_dir) for entry in MANIFEST.values())
    if any(same_file(path, value) for value in protected):
        raise ValueError('Output must be a different file from every input and original checkpoint')
    return path


def setup_weights(weights_dir=None):
    root = Path(weights_dir or default_weights_dir()).expanduser()
    for name, entry in MANIFEST.items():
        target = root / entry['file']
        if target.exists():
            verify_baseline(name, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                urllib.request.urlretrieve(entry['url'], temporary)
                verify_baseline(name, temporary)
                # Atomic publication without replacing a file created concurrently.
                temporary.chmod(0o444)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        target.chmod(target.stat().st_mode & ~0o222)
        print(f'Verified read-only {name}: {target}')
