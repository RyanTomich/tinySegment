"""Restore the pinned detector source before installing an editable checkout or building a wheel."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve
from zipfile import ZipFile

UPSTREAM = Path(__file__).resolve().parent / 'upstream'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(directory, hashes):
    for relative, expected in hashes.items():
        path = directory / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or modified Ultralytics source: {path}. Move the source directory aside to restore it.')


def restore():
    package = UPSTREAM / 'ultralytics'
    source = json.loads((UPSTREAM / 'provenance.json').read_text())['ultralytics']
    recorded = json.loads((UPSTREAM / 'source-hashes.json').read_text())
    # The bus image is retained separately as the tracked integration fixture.
    hashes = {name.removeprefix('ultralytics/'): value for name, value in recorded.items()
              if name.startswith('ultralytics/') and name != 'ultralytics/assets/bus.jpg'}
    if package.exists():
        verify(package, hashes)
        return package

    with tempfile.TemporaryDirectory(prefix='.build-', dir=UPSTREAM) as temp:
        archive = Path(temp) / 'release.whl'
        urlretrieve(source['wheel']['url'], archive)
        if digest(archive) != source['wheel']['sha256']:
            raise ValueError('Downloaded Ultralytics wheel failed its SHA-256 check')
        staged = Path(temp) / 'ultralytics'
        with ZipFile(archive) as wheel:
            for name in hashes:
                member = f'ultralytics/{name}'
                if name == 'LICENSE':
                    member = f'ultralytics-{source["version"]}.dist-info/licenses/LICENSE'
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with wheel.open(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        verify(staged, hashes)
        try:
            staged.rename(package)
        except OSError:
            if not package.is_dir():
                raise
            verify(package, hashes)
    return package


if __name__ == '__main__':
    print(f'Verified detector source: {restore()}')
