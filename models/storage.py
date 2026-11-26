"""Local checkpoints, baseline integrity, and protected artifact destinations."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

MODEL_SOURCE = Path(__file__).resolve().parent
FAMILIES = {'clip': 'clip_vit_b32', 'detector': 'yolo_world_v2_small',
            'segmenter': 'efficientvit_sam_l0', 'speech': 'moonshine_tiny_streaming_en'}


def manifest_entries(family):
    baseline = MODEL_SOURCE / family / 'baseline'
    entries = json.loads((baseline / 'manifest.json').read_text())
    if isinstance(entries, dict):
        entries = [entries]
    return [dict(entry, file=str(Path(family) / 'baseline' / 'artifacts' / entry['file']))
            for entry in entries]


MANIFEST = {name: manifest_entries(family)[0] for name, family in FAMILIES.items() if name != 'speech'}


def project_weights_dir():
    if (MODEL_SOURCE.parent / 'pyproject.toml').is_file():
        return MODEL_SOURCE
    return Path.home() / '.cache' / 'tinysegment' / 'models'


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
    model_extensions = {'.pt', '.pth', '.ckpt', '.safetensors', '.pte', '.ort', '.onnx'}
    if path.suffix.lower() in model_extensions or resolved.suffix.lower() in model_extensions:
        raise ValueError(f'Prediction output cannot overwrite a model checkpoint: {path}')
    protected = [value for value in inputs if value is not None]
    entries = [*MANIFEST.values(), *manifest_entries(FAMILIES['speech'])]
    protected.extend(root / entry['file'] for root in baseline_roots(weights_dir) for entry in entries)
    if any(same_file(path, value) for value in protected):
        raise ValueError('Output must be a different file from every input and original checkpoint')
    return path


def setup_weights(weights_dir=None, *, speech=False):
    root = Path(weights_dir or default_weights_dir()).expanduser()
    entries = list(MANIFEST.values())
    if speech:
        entries.extend(manifest_entries(FAMILIES['speech']))
    for entry in entries:
        target = root / entry['file']
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                urllib.request.urlretrieve(entry['url'], temporary)
                verify_download(temporary, entry)
                temporary.chmod(0o444)
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass  # Verify the concurrent publication below.
            finally:
                temporary.unlink(missing_ok=True)
        verify_download(target, entry)
        target.chmod(target.stat().st_mode & ~0o222)
        print(f'Verified {target}')


def verify_download(path, entry):
    if sha256(path) != entry['sha256']:
        raise ValueError(f"Baseline checksum mismatch for {entry['file']}: {path}")


def variant_artifacts(variant, weights_dir=None):
    return Path(weights_dir or default_weights_dir()).expanduser() / FAMILIES['segmenter'] / variant / 'artifacts'
