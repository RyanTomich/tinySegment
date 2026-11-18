"""Exclusive run directories and atomic, never-overwriting artifact writes."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / 'runs' / 'sam'


def reject_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlinks are not allowed in research output paths: {part}')


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    path: Path

    def destination(self, name):
        reject_symlinks(self.root)
        reject_symlinks(self.path)
        if self.path.resolve().parent != self.root.resolve():
            raise ValueError('Run directory must be directly inside the runs root')
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Run artifacts must be plain filenames')
        target = self.path / name
        reject_symlinks(target)
        if target.exists():
            raise FileExistsError(f'Refusing to replace an existing run artifact: {target}')
        return target

    def write(self, name, writer):
        target = self.destination(name)
        with tempfile.NamedTemporaryFile(dir=self.path, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            writer(temporary)
            # Recheck after serialization, then publish with no-clobber semantics.
            self.destination(name)
            temporary.chmod(0o444)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def json(self, name, data):
        return self.write(name, lambda path: path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'))

    def text(self, name, text):
        return self.write(name, lambda path: path.write_text(text))

    def checkpoint(self, model):
        import torch
        state = model.state_dict()
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError('Only tensor state dictionaries compatible with eager L0 are supported')
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Checkpoint contains nonfinite weights')
        state = {name: value.detach().cpu() for name, value in state.items()}
        def serialize(path):
            with path.open('wb') as stream:
                torch.save(state, stream)

        return self.write('checkpoint.pt', serialize)


def allocate_run(name, runs_root=RUNS_ROOT):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name):
        raise ValueError('Run name must be 1–64 letters, digits, underscores, or hyphens, starting with a letter or digit')
    root = Path(runs_root).absolute()
    from tinysegment.weights import baseline_roots
    if any(root.resolve() == reserved or reserved in root.resolve().parents for reserved in baseline_roots()):
        raise ValueError('Research runs cannot be created inside original weights storage')
    reject_symlinks(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = root / f'{stamp}-{name}-{uuid.uuid4().hex[:8]}'
    path.mkdir(exist_ok=False)
    return RunDirectory(root, path)
