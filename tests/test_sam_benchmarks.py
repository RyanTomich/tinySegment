"""SAM benchmark contracts with tiny deterministic inputs."""
import importlib.util
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from research.sam.benchmark import benchmark, paired_samples
from research.sam.quality import MANIFEST, mask_iou, quality


class BenchmarkTests(unittest.TestCase):
    def test_manifest_and_iou(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(len(manifest), 100)
        ids = list(itertools.chain.from_iterable(manifest.values()))
        self.assertEqual(len(ids), len(set(ids)))
        a = np.array([True, True, False])
        b = np.array([False, True, True])
        self.assertAlmostEqual(mask_iou(a, b), 1 / 3)
        self.assertEqual(mask_iou(a & False, b & False), 1)

    def test_paired_clock(self):
        order = []
        clock = itertools.count(0, 2_000_000).__next__
        samples = paired_samples([lambda: order.append(0), lambda: order.append(1)], 1, 2, clock)
        self.assertEqual(order, [0, 1, 0, 1, 1, 0])
        self.assertEqual(samples['candidate_ms'], [2, 2])
        self.assertEqual(samples['median_ms']['original'], 2)
        self.assertEqual(samples['p95_ms']['delta_pct'], 0)

    @unittest.skipUnless(importlib.util.find_spec('fvcore'), 'requires fvcore')
    def test_dense_macs(self):
        from research.sam.size_ops import count_macs
        macs, coverage = count_macs(torch.nn.Linear(3, 2), (torch.ones(4, 3),))
        self.assertEqual(macs, 24)
        self.assertTrue(coverage['complete'])

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'), 'requires pycocotools')
    def test_coco_boxes_and_aggregation(self):
        class Predictor:
            def __init__(self, empty):
                self.empty = empty
                self.images = 0
                self.boxes = []

            def set_image(self, image):
                self.images += 1

            def predict(self, *, box, multimask_output):
                self.boxes.append(box.tolist())
                mask = np.zeros((1, 6, 6), dtype=bool)
                x, y, end_x, end_y = box.astype(int)
                mask[:, y:end_y, x:end_x] = not self.empty
                return mask, np.ones(1), None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'annotations').mkdir()
            (root / 'val2017').mkdir()
            Image.new('RGB', (6, 6)).save(root / 'val2017/test.jpg')
            anns = [dict(id=i, image_id=1, category_id=1, area=4, bbox=[x, 1, 2, 2], iscrowd=int(i == 3),
                         segmentation=[[x, 1, x+2, 1, x+2, 3, x, 3]]) for i, x in [(1, 1), (2, 3), (3, 2)]]
            data = dict(images=[dict(id=1, width=6, height=6, file_name='test.jpg')], annotations=anns)
            (root / 'annotations/instances_val2017.json').write_text(json.dumps(data))
            stages = [SimpleNamespace(predictor=Predictor(empty)) for empty in [False, True]]
            result = quality(*stages, root, full=True)
            self.assertEqual(result['groups']['all'], dict(instances=2, original_miou_pct=100,
                             candidate_miou_pct=0, delta_points=-100, fidelity_miou_pct=0))
            self.assertEqual(result['groups']['small'], result['groups']['all'])
            self.assertEqual(result['groups']['large'], {'instances': 0})
            for stage in stages:
                self.assertEqual(stage.predictor.images, 1)
                self.assertEqual(stage.predictor.boxes, [[1, 1, 3, 3], [3, 1, 5, 3]])

    def test_independent_reports_and_failures(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = root / 'example'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps({'original_checkpoint': 'baseline.pt'}))
            (run / 'alternate.pt').write_bytes(b'example')
            with patch('research.sam.evaluate.RUNS_ROOT', root), patch('research.sam.evaluate.verify_baseline', return_value='test-hash'), \
                 patch('research.sam.benchmark.load_segmenter'), patch('research.sam.benchmark.latency', return_value={}):
                first = benchmark('latency', run, 'alternate.pt', warmup=1, samples=2)
                previous = first.read_bytes()
                second = benchmark('latency', run, 'alternate.pt', warmup=1, samples=2)
                self.assertNotEqual(first, second)
                with self.assertRaises(FileNotFoundError):
                    benchmark('latency', run, 'missing.pt', warmup=1, samples=2)
                self.assertEqual(first.read_bytes(), previous)
                reports = [json.loads(p.read_text()) for p in run.glob('evaluations/*/report.json')]
                self.assertCountEqual([r['status'] for r in reports], ['success', 'success', 'failed'])
