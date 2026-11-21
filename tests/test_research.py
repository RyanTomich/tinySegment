"""Safety contracts and real-checkpoint SAM research integration."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from research.sam.checkpoints import allocate_run
from research.sam.evaluate import evaluate
from research.sam.run import execute, PROJECT_ROOT
from tinysegment.weights import (
    MANIFEST, checkpoint_path, sha256, setup_weights, validate_output_path,
)


class ResearchSafety(unittest.TestCase):
    def test_evaluator_import_is_independent(self):
        subprocess.run([sys.executable, '-c',
                        "import sys; sys.modules['research.sam.run'] = sys.modules['research.sam.recipes'] = None; "
                        "from research.sam.evaluate import evaluate"], cwd=PROJECT_ROOT, check=True)

    def test_unique_runs_and_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            first, second = allocate_run('baseline', root), allocate_run('baseline', root)
            self.assertNotEqual(first.path, second.path)
            model = torch.nn.Linear(2, 1)
            checkpoint = first.checkpoint(model)
            before = sha256(checkpoint)
            with self.assertRaises(FileExistsError): first.checkpoint(model)
            self.assertEqual(sha256(checkpoint), before)
            self.assertEqual(checkpoint.stat().st_mode & 0o222, 0)
            self.assertFalse(list(first.path.glob('.pending-*')))
            with self.assertRaises(ValueError): first.text('../escape', 'invalid')

    def test_original_storage_cannot_be_a_run_root(self):
        with self.assertRaises(ValueError):
            allocate_run('unsafe', PROJECT_ROOT / 'weights' / 'experiments')

    def test_collision_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('research.sam.checkpoints.datetime') as clock, patch('research.sam.checkpoints.uuid.uuid4') as uuid:
                clock.now.return_value.strftime.return_value = 'fixed'
                uuid.return_value.hex = '12345678'
                allocate_run('baseline', Path(directory).resolve())
                with self.assertRaises(FileExistsError): allocate_run('baseline', Path(directory).resolve())

    def test_symlinks_and_failed_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / 'outside'; outside.mkdir()
            link = root / 'link'; link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError): allocate_run('unsafe', link)
            run = allocate_run('safe', root / 'runs')
            (run.path / 'checkpoint.pt').symlink_to(outside / 'target.pt')
            with self.assertRaises(ValueError): run.checkpoint(torch.nn.Linear(2, 1))
            self.assertFalse((outside / 'target.pt').exists())
            def broken(path):
                path.write_bytes(b'partial')
                raise RuntimeError('interrupted serialization')
            with self.assertRaises(RuntimeError): run.write('other.pt', broken)
            self.assertFalse((run.path / 'other.pt').exists())
            self.assertFalse(list(run.path.glob('.pending-*')))

    def test_output_protection_including_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            weights = root / 'weights'; (weights / 'sam').mkdir(parents=True)
            original = weights / MANIFEST['segmenter']['file']; original.write_bytes(b'original')
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
            with patch('tinysegment.weights.MANIFEST', manifest):
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


@unittest.skipUnless(os.environ.get('TINYSEGMENT_INTEGRATION') == '1', 'set TINYSEGMENT_INTEGRATION=1 for real weights')
class ResearchIntegration(unittest.TestCase):
    def test_baseline_candidate_and_failure_lifecycle(self):
        from tinysegment import load_segmenter
        from tinysegment.pipeline import read_image, run
        originals = {name: checkpoint_path(name) for name in MANIFEST}
        before = {name: sha256(path) for name, path in originals.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            def allocate(name): return allocate_run(name, root)
            with patch('research.sam.run.allocate_run', side_effect=allocate), patch('research.sam.evaluate.RUNS_ROOT', root):
                with patch('tinysegment.models.Segmenter.segment', side_effect=AssertionError('inference during generation')), \
                     patch('tinysegment.pipeline.read_image', side_effect=AssertionError('fixture during generation')):
                    result = execute()
                self.assertFalse((result / 'evaluations').exists())
                self.assertFalse((result / 'report.json').exists())
                with patch.dict(sys.modules, {'research.sam.run': None, 'research.sam.recipes': None}):
                    first = evaluate(result)
                    first_bytes = first.read_bytes()
                    second_report = evaluate(result)
                self.assertNotEqual(first, second_report)
                self.assertEqual(first.read_bytes(), first_bytes)
                report = json.loads(first_bytes)
                self.assertEqual(report['status'], 'success')
                self.assertEqual(report['baseline_fidelity']['mean_mask_iou'], 1)
                candidate = result / 'checkpoint.pt'
                self.assertEqual(sha256(candidate), report['output_sha256'])
                self.assertEqual(load_segmenter().checkpoint, originals['segmenter'])
                image = read_image(PROJECT_ROOT / 'vendor/ultralytics/assets/bus.jpg')
                detections, masks, overlay = run(image, ['person'], segmenter_checkpoint=candidate)
                self.assertGreater(len(detections['boxes']), 0)
                self.assertTrue(masks['combined_mask'].any())
                self.assertEqual(overlay.size, image.size)
                # Starting from a prior run is explicit and never updates it.
                second = execute(name='baseline', input_checkpoint=candidate)
                self.assertNotEqual(second, result)
                self.assertEqual(json.loads((second / 'metadata.json').read_text())['input_sha256'], report['output_sha256'])
                self.assertEqual(sha256(candidate), report['output_sha256'])

                alternate = result / 'alternate.pt'
                os.link(candidate, alternate)
                alternate_report = json.loads(evaluate(result, checkpoint=alternate.name).read_text())
                self.assertEqual(alternate_report['checkpoint'], alternate.name)
                self.assertEqual(alternate_report['output_sha256'], report['output_sha256'])

                (result / 'invalid.pt').write_bytes(b'invalid')
                with self.assertRaises(ValueError):
                    evaluate(result, checkpoint='invalid.pt')
                reports = [json.loads(p.read_text()) for p in result.glob('evaluations/*/report.json')]
                self.assertEqual([r['status'] for r in reports if r['checkpoint'] == 'invalid.pt'], ['failed'])
                self.assertEqual(first.read_bytes(), first_bytes)
                self.assertEqual(sha256(candidate), report['output_sha256'])

                def interrupted(model, config):
                    raise KeyboardInterrupt()
                with patch.dict('research.sam.run.RECIPES', {'interrupted': interrupted}):
                    with self.assertRaises(KeyboardInterrupt):
                        execute(name='interrupted', recipe='interrupted')
                failed = next(root.glob('*-interrupted-*'))
                metadata = json.loads((failed / 'metadata.json').read_text())
                self.assertEqual(metadata['status'], 'interrupted')
                self.assertTrue(metadata['original_unchanged'])
            self.assertEqual(before, {name: sha256(path) for name, path in originals.items()})
