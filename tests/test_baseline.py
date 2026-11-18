"""Run with: python -m unittest discover -s tests -v.
Set TINYSEGMENT_INTEGRATION=1 to run real-weight isolated CLI checks too.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from tinysegment.models import Detector, Segmenter, TextEncoder
from tinysegment.pipeline import read_image, render_overlay
from tinysegment.weights import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def test_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'setup-weights'):
                checkpoint_path('clip', directory)

    def test_empty_prompt(self):
        stage = object.__new__(TextEncoder)
        with self.assertRaises(ValueError): stage.tokenize([' '])

    def test_empty_masks_skip_image_encoder(self):
        stage = object.__new__(Segmenter)
        result = stage.segment(Image.new('RGB', (13, 17)), [])
        self.assertEqual(result['masks'].shape, (0, 17, 13))
        self.assertFalse(result['combined_mask'].any())

    def test_invalid_boxes(self):
        stage = object.__new__(Segmenter)
        for boxes in [[[1,2,0,4]], [[0,0,float('nan'),4]], [1,2,3,4]]:
            with self.assertRaises(ValueError): stage.segment(Image.new('RGB',(10,10)), boxes)

    def test_no_detections_overlay(self):
        image = Image.new('RGB',(13,17),(12,34,56))
        overlay = render_overlay(image, [], np.zeros((17,13),bool))
        np.testing.assert_array_equal(overlay, image.convert('RGBA'))

    def test_exif_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.jpg'
            image = Image.new('RGB',(13,17)); exif = image.getexif(); exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_image(path).size, (17,13))


# Each subprocess rejects network connections and file access to sibling projects.
# This proves independence without renaming or modifying another repository.
ISOLATED = r'''
import os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
command = sys.argv[2:]
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network disabled by independence test')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        if root.parent in path.parents and root not in path.parents and path != root:
            raise RuntimeError('Sibling project access blocked: '+str(path))
sys.addaudithook(audit)
from tinysegment.cli import main
main(command)
if command[0] == 'detect':
    assert 'clip' not in sys.modules and 'efficientvit' not in sys.modules
if command[0] == 'segment':
    assert 'clip' not in sys.modules and 'ultralytics' not in sys.modules
for name in ('clip', 'ultralytics', 'efficientvit', 'segment_anything'):
    module = sys.modules.get(name)
    if module is not None:
        assert root / 'vendor' in Path(module.__file__).resolve().parents, module.__file__
'''


@unittest.skipUnless(os.environ.get('TINYSEGMENT_INTEGRATION') == '1', 'set TINYSEGMENT_INTEGRATION=1 for real checkpoints')
class Integration(unittest.TestCase):
    def cli(self, *args, expected=0):
        environment = dict(os.environ, YOLO_CONFIG_DIR=str(ROOT/'.runtime/tests'), MPLCONFIGDIR=str(ROOT/'.runtime/matplotlib'))
        result = subprocess.run([sys.executable, '-c', ISOLATED, str(ROOT), *map(str,args)], cwd=tempfile.gettempdir(), env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, expected, result.stdout+'\n'+result.stderr)
        return result

    def test_independent_stages_and_pipeline(self):
        image = ROOT/'vendor/ultralytics/assets/bus.jpg'
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            self.cli('encode-text','person','bus','--output',out/'text.npz')
            with np.load(out/'text.npz') as data:
                self.assertEqual(data['embeddings'].shape,(1,2,512))
                np.testing.assert_allclose(np.linalg.norm(data['embeddings'],axis=-1),1,atol=1e-6)
            self.cli('detect',image,'--embeddings',out/'text.npz','--output',out/'boxes.json')
            detections=json.loads((out/'boxes.json').read_text())
            self.assertGreater(len(detections['boxes']),1)
            self.cli('segment',image,'--boxes',out/'boxes.json','--output',out/'masks.npz')
            with np.load(out/'masks.npz') as masks:
                self.assertEqual(masks['masks'].shape,(len(detections['boxes']),1080,810))
                self.assertTrue(masks['combined_mask'].any())
            for flag in [[],['--legacy-marked-input']]:
                self.cli('run',image,'person','--output',out/'overlay.png',*flag)
                with Image.open(out/'overlay.png') as overlay: self.assertEqual(overlay.size,(810,1080))
            self.cli('run',image,'person','--confidence','1','--output',out/'empty.png')
            np.testing.assert_array_equal(np.asarray(Image.open(out/'empty.png')),np.asarray(read_image(image).convert('RGBA')))
            self.cli('run',image,' ','--output',out/'invalid.png',expected=2)
            self.cli('run',image,'person','--output',image,expected=2)
            self.cli('run',out/'missing.jpg','person','--output',out/'invalid.png',expected=2)
            bad=out/'bad.pt'; bad.write_bytes(b'not a checkpoint')
            for model in ['clip','detector','segmenter']:
                self.cli('inspect',model,'--checkpoint',bad,expected=2)

    def test_model_access_and_state_roundtrip(self):
        import torch
        from tinysegment import load_text_encoder,load_detector,load_segmenter
        for loader in [load_text_encoder,load_detector,load_segmenter]:
            stage=loader()
            self.assertIsInstance(stage.model,torch.nn.Module)
            self.assertFalse(stage.model.training)
            self.assertGreater(len(list(stage.model.named_modules())),10)
            name, child=next(iter(stage.model.named_children()))
            setattr(stage.model,name,torch.nn.Identity())
            self.assertIsInstance(getattr(stage.model,name),torch.nn.Identity)
            setattr(stage.model,name,child)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'state.pt'
                torch.save(stage.model.state_dict(),path)
                result=stage.model.load_state_dict(torch.load(path,weights_only=True),strict=True)
                self.assertEqual(result.missing_keys,[])
                self.assertEqual(result.unexpected_keys,[])
            del stage
