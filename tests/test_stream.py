import unittest
from threading import Event
from unittest.mock import Mock, patch

from tinysegment.cli import latest_prompts
from tinysegment.pipeline import Pipeline


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

    def test_eager_stages_reused_for_repeated_prompts(self):
        text, detector, sam = Mock(), Mock(), Mock()
        with patch('tinysegment.pipeline.load_stage', side_effect=[text, detector, sam]) as load:
            pipeline = Pipeline()
            self.assertEqual([call.args for call in load.call_args_list],
                             [('clip', 'baseline'), ('detector', 'baseline'), ('segmenter', 'baseline')])
            detector.detect.return_value = {'boxes': []}
            for prompt in ('person', 'person', 'bus'):
                pipeline.process(object(), [prompt])
            self.assertEqual(text.encode.call_count, 2)
            self.assertEqual(detector.detect.call_count, 3)
            self.assertEqual(sam.segment.call_count, 3)
            self.assertEqual(load.call_count, 3)
