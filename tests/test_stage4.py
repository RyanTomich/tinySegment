"""Contracts for the stage 4 experiment's interleaved timing and paired decision."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from efficientvit.models.nn.ops import OpSequential
from torch import nn

from research.sam.profile_resident import trace_backbone
from research.sam.stage4 import samples, saving
from research.sam.model import RuntimeBlock, load_research_segmenter


class Stage4TimingTests(unittest.TestCase):
    def test_rotated_execution_preserves_round_pairing(self):
        order = []
        clock = [0]

        def call(name, duration):
            order.append(name)
            clock[0] += duration

        calls = {name: (lambda n=name, d=ms: call(n, d * 1_000_000))
                 for name, ms in [('eager', 10), ('fp32', 5), ('int8', 3)]}
        with patch('research.sam.stage4.perf_counter_ns', side_effect=lambda: clock[0]):
            result = samples(calls, 3)
        self.assertEqual(order[:9], list(calls) * 3)
        self.assertEqual(order[9:], ['eager', 'fp32', 'int8', 'fp32', 'int8', 'eager',
                                     'int8', 'eager', 'fp32'])
        self.assertEqual(result, {'eager': [10] * 3, 'fp32': [5] * 3, 'int8': [3] * 3})

    def test_bootstrap_keeps_pairs_and_direction_of_savings(self):
        self.assertEqual(saving([90, 31, 205], [85, 26, 200]),
                         {'median_ms': 5.0, 'ci95_ms': [5.0, 5.0]})
        self.assertEqual(saving([4, 8, 6], [7, 11, 9])['ci95_ms'], [-3.0, -3.0])

    def test_research_loader_requires_both_programs(self):
        for missing, present in ((3, 4), (4, 3)):
            with self.subTest(stage=missing), TemporaryDirectory() as tmp, \
                    patch('research.sam.model.PROGRAM_DIR', Path(tmp)):
                (Path(tmp) / f'stage{present}-int8.pte').touch()
                with self.assertRaisesRegex(FileNotFoundError, f'stage{missing}-int8'):
                    load_research_segmenter()

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
            with patch('research.sam.model.PROGRAM_DIR', Path(tmp).resolve()), \
                    patch('research.sam.model.load_segmenter', return_value=stage), \
                    patch('research.sam.model.RuntimeBlock', side_effect=programs.__getitem__):
                load_research_segmenter()
            for i, path in paths.items():
                self.assertEqual(stage.metadata[f'stage{i}_program']['path'], str(path.resolve()))
        self.assertEqual([list(s.op_list) for s in stages], expected)

    def test_profile_times_compiled_stage3_entry(self):
        compiled = RuntimeBlock.__new__(RuntimeBlock)
        nn.Module.__init__(compiled)
        compiled.method = Mock(execute=lambda inputs: inputs)
        stages = [OpSequential([nn.Identity()]) for _ in range(5)]
        stages[3].op_list[0] = compiled
        times = {}
        with ExitStack() as stack:
            trace_backbone(stack, times, SimpleNamespace(stages=stages))
            self.assertEqual(stages[3](42), 42)
        self.assertEqual(set(times), {'stage_3', 'stage_3_block_0'})
