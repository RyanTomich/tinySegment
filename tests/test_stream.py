from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from threading import Event
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image

from tinysegment.cli import main
from tinysegment.pipeline import latest_prompts, segment_image


class StreamTests(unittest.TestCase):
    def test_only_latest_pending_prompt_survives(self):
        first_received = Event()
        burst_read = Event()

        def lines():
            yield 'first'
            first_received.wait(5)
            yield 'second'
            yield 'third'
            yield '  '
            burst_read.set()

        prompts = iter(latest_prompts(lines()))
        self.assertEqual(next(prompts), 'first')
        first_received.set()
        self.assertTrue(burst_read.wait(5))
        self.assertEqual(list(prompts), ['third'])

    def test_last_prompt_is_delivered_before_stdin_error(self):
        def lines():
            yield 'bus'
            raise OSError('stdin disconnected')

        prompts = iter(latest_prompts(lines()))
        self.assertEqual(next(prompts), 'bus')
        with self.assertRaisesRegex(OSError, 'stdin disconnected'):
            next(prompts)

    def test_eager_stages_reused_for_repeated_prompts(self):
        text, detector, sam = Mock(), Mock(), Mock()
        person, bus = object(), object()
        text.encode.side_effect = [person, bus]
        detector.detect.return_value = {'boxes': []}
        sam.segment.return_value = {'combined_mask': np.zeros((2, 3), bool)}
        with TemporaryDirectory() as directory:
            image = Path(directory) / 'frame.png'
            Image.new('RGB', (3, 2)).save(image)
            with (
                patch('tinysegment.pipeline.load_stage', side_effect=[text, detector, sam]) as load,
                patch('tinysegment.cli.latest_prompts', return_value=['person', 'person', 'bus']),
            ):
                main(['run', str(image), '--stream', '--output', str(Path(directory) / 'mask.png')])
        self.assertEqual([c.args for c in load.call_args_list],
                         [('clip', 'baseline'), ('detector', 'baseline'), ('segmenter', 'baseline')])
        self.assertEqual(text.encode.call_args_list, [call(('person',)), call(('bus',))])
        self.assertEqual([c.args[1:3] for c in detector.detect.call_args_list],
                         [(person, ('person',)), (person, ('person',)), (bus, ('bus',))])
        self.assertEqual(sam.segment.call_count, 3)

    def test_segment_image_preserves_box_order_and_only_marks_legacy_input(self):
        image = Image.new('RGB', (10, 10))
        embeddings = object()
        boxes = [[1, 2, 5, 8], [0, 0, 3, 4]]
        detector, sam = Mock(), Mock()
        detector.detect.return_value = {'boxes': boxes}
        result = segment_image(image, ('bus',), embeddings, detector, sam)
        self.assertEqual(result, (detector.detect.return_value, sam.segment.return_value))
        detector.detect.assert_called_once_with(image, embeddings, ('bus',), .25, .7, 640)
        sam.segment.assert_called_once_with(image, boxes)

        segment_image(image, ('bus',), embeddings, detector, sam, legacy_marked_input=True)
        marked, received_boxes = sam.segment.call_args.args
        self.assertEqual(received_boxes, boxes)
        self.assertEqual(marked.getpixel((3, 5)), (0, 235, 65))
        self.assertEqual(image.getpixel((3, 5)), (0, 0, 0))
