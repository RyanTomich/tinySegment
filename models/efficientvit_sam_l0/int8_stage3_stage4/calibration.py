"""Pinned COCO images for calibrating the stage 3 and 4 entry blocks."""
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlretrieve

from models.storage import sha256


# First 16 eligible COCO val2017 images outside the fixed quality evaluation.
CALIBRATION = json.loads(Path(__file__).with_name('calibration.json').read_text())


def calibration_paths(root):
    directory = root / 'val2017'
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in CALIBRATION.items():
        path = directory / name
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
                urlretrieve(f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{path.name}', temp.name)
                if sha256(temp.name) != expected:
                    raise ValueError(f'Calibration checksum mismatch: {name}')
                try:
                    os.link(temp.name, path)
                except FileExistsError:
                    pass  # Another run published the image first.
        if sha256(path) != expected:
            raise ValueError(f'Calibration checksum mismatch: {path}')
        paths.append(path)
    return paths
