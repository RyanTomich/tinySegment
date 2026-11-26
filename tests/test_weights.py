"""Checkpoint setup and output protection contracts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.storage import (
    MANIFEST, checkpoint_path, sha256, setup_weights, validate_output_path,
)


class WeightSafetyTests(unittest.TestCase):
    def test_output_protection_including_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            weights = root / 'weights'
            original = weights / MANIFEST['segmenter']['file']
            original.parent.mkdir(parents=True)
            original.write_bytes(b'original')
            candidate = root / 'candidate.bin'; candidate.write_bytes(b'candidate')
            alias = root / 'alias.png'; alias.symlink_to(original)
            hardlink = root / 'hardlink.png'; os.link(original, hardlink)
            for output in [weights / 'new.png', original, alias, hardlink, root / 'unknown.pt', candidate]:
                with self.assertRaises(ValueError):
                    validate_output_path(output, weights_dir=weights, inputs=[candidate])
            self.assertEqual(original.read_bytes(), b'original')
            validate_output_path(root / 'output.png', weights_dir=weights, inputs=[candidate])

    def test_mismatch_and_readonly_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / 'test.pt'; path.write_bytes(b'baseline')
            manifest = {'segmenter': {'file': 'test.pt', 'sha256': sha256(path)}}
            with patch('models.storage.MANIFEST', manifest):
                setup_weights(root)
                self.assertEqual(path.stat().st_mode & 0o222, 0)
                self.assertEqual(checkpoint_path('segmenter', weights_dir=root), path)
                path.chmod(0o644); path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    checkpoint_path('segmenter', weights_dir=root)
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    checkpoint_path('segmenter', weights_dir=root, checkpoint=path)
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'): setup_weights(root)
                self.assertEqual(path.read_bytes(), b'changed')

    def test_speech_download_is_explicit_and_rejects_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'speech')
            target = root / 'models'
            entry = {'file': 'moonshine_tiny_streaming_en/baseline/artifacts/test.ort',
                     'url': source.as_uri(), 'sha256': sha256(source)}
            with patch('models.storage.MANIFEST', {}), patch('models.storage.manifest_entries', return_value=[entry]):
                setup_weights(target)
                self.assertFalse(target.exists())
                source.write_bytes(b'truncated')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    setup_weights(target, speech=True)
                self.assertFalse((target / entry['file']).exists())
                source.write_bytes(b'speech')
                setup_weights(target, speech=True)
                self.assertEqual((target / entry['file']).read_bytes(), b'speech')

    def test_cli_rejects_checkpoint_destination_before_model_load(self):
        from tinysegment.cli import main
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'experiment.bin'; checkpoint.write_bytes(b'candidate')
            with patch('tinysegment.cli.load_stage') as loader:
                with self.assertRaises(SystemExit) as caught:
                    main(['encode-text', 'person', '--checkpoint', str(checkpoint), '--output', str(checkpoint)])
                self.assertEqual(caught.exception.code, 2)
                loader.assert_not_called()
                self.assertEqual(checkpoint.read_bytes(), b'candidate')
