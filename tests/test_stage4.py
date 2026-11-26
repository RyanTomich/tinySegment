"""Loader and selection contracts for the INT8 stage 3 and 4 variant."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficientvit.models.nn.ops import OpSequential
from torch import nn

from models.efficientvit_sam_l0.int8_stage3_stage4 import loader as variant
from models.efficientvit_sam_l0.int8_stage3_stage4.loader import load_research_segmenter
from tinysegment.pipeline import load_stage


class StageLoaderTests(unittest.TestCase):
    def test_default_programs_follow_model_storage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = root / 'efficientvit_sam_l0/int8_stage3_stage4/artifacts'
            programs = variant.program_paths(weights_dir=root)
            self.assertEqual({p.parent for p in programs.values()}, {expected})
            self.assertFalse(list(root.iterdir()))

    def test_research_loader_requires_both_programs(self):
        for missing, present in ((3, 4), (4, 3)):
            with self.subTest(stage=missing), TemporaryDirectory() as tmp:
                (Path(tmp) / f'stage{present}-int8.pte').touch()
                with self.assertRaisesRegex(FileNotFoundError, f'stage{missing}-int8'):
                    load_research_segmenter(program_dir=tmp)

    def test_combined_loader_installs_each_program_in_its_own_entry(self):
        stages = [OpSequential([nn.Identity(), nn.Identity()]) for _ in range(5)]
        stage = SimpleNamespace(model=SimpleNamespace(image_encoder=SimpleNamespace(
            backbone=SimpleNamespace(stages=stages))), metadata={})
        expected = [list(s.op_list) for s in stages]
        with TemporaryDirectory() as tmp:
            paths = {i: Path(tmp).resolve() / f'stage{i}-int8.pte' for i in (3, 4)}
            programs = {path: nn.Identity() for path in paths.values()}
            for i, path in paths.items():
                path.write_bytes(f'stage {i}'.encode())
                expected[i][0] = programs[path]
            with patch.object(variant, 'load_segmenter', return_value=stage), \
                    patch.object(variant, 'RuntimeBlock', side_effect=programs.__getitem__):
                selected = load_stage('segmenter', 'int8_stage3_stage4', program_dir=tmp,
                                      checkpoint='custom.pth')
                self.assertIs(selected, stage)
                variant.load_segmenter.assert_called_once_with(weights_dir=None, checkpoint='custom.pth',
                                                                device='cpu')
            for i, path in paths.items():
                self.assertEqual(stage.metadata[f'stage{i}_program']['path'], str(path.resolve()))
        self.assertEqual([list(s.op_list) for s in stages], expected)

    def test_int8_requires_cpu(self):
        with self.assertRaisesRegex(ValueError, 'CPU'):
            load_stage('segmenter', 'int8_stage3_stage4', device='mps')

    def test_rejects_variant_for_wrong_stage(self):
        with self.assertRaisesRegex(ValueError, 'detector'):
            load_stage('detector', 'int8_stage3_stage4')
