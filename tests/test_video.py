from pathlib import Path
import runpy
import sys
import unittest
from queue import Queue
from threading import Event, current_thread, main_thread
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np

from tinysegment.pipeline import run, show_mask


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.startup = []
        self.text, self.detector, self.sam = Mock(), Mock(), Mock()
        self.detector.detect.return_value = {'boxes': [[0, 0, 2, 1]]}
        self.stages = {'clip': self.text, 'detector': self.detector, 'segmenter': self.sam}

        def load_stage(stage, variant, **options):
            self.startup.append(stage)
            return self.stages[stage]

        self.enterContext(patch('tinysegment.pipeline.load_stage', side_effect=load_stage))
        self.transcriber = MagicMock()

        def load_speech(**options):
            self.startup.append('speech')
            return self.transcriber

        speech = 'models.moonshine_tiny_streaming_en.baseline.loader'
        self.load_speech = self.enterContext(patch(f'{speech}.load_transcriber', side_effect=load_speech))
        self.transcribe = self.enterContext(patch(f'{speech}.completed_lines', return_value=()))

        self.camera = Mock()

        def open_camera(*args):
            self.startup.append('camera')
            return self.camera

        self.open_camera = self.enterContext(patch('cv2.VideoCapture', side_effect=open_camera))
        self.device = Mock()
        self.device.PortAudioError = OSError
        self.device.query_devices.return_value = {'default_samplerate': 48000}
        self.mic = MagicMock()

        def open_mic(**options):
            self.startup.append('microphone')
            return self.mic

        self.device.InputStream.side_effect = open_mic
        self.enterContext(patch.dict(sys.modules, sounddevice=self.device))
        self.open_window = self.enterContext(patch('cv2.namedWindow'))
        self.close_window = self.enterContext(patch('cv2.destroyAllWindows'))
        self.show = self.enterContext(patch('tinysegment.pipeline.show_mask', return_value=True))

    def test_direct_script_starts_camera_zero_and_waits_for_speech(self):
        for module, loader, stage in (
            ('clip_vit_b32', 'load_text_encoder', self.text),
            ('yolo_world_v2_small', 'load_detector', self.detector),
            ('efficientvit_sam_l0', 'load_segmenter', self.sam),
        ):
            self.enterContext(patch(f'models.{module}.baseline.loader.{loader}', return_value=stage))
        self.camera.read.return_value = True, np.zeros((2, 3, 3), dtype=np.uint8)
        display = self.enterContext(patch('cv2.imshow'))
        self.enterContext(patch('cv2.waitKey', return_value=ord('q')))
        script = Path(__file__).resolve().parents[1] / 'src/tinysegment/pipeline.py'
        runpy.run_path(str(script), run_name='__main__')

        self.assertEqual(self.open_camera.call_args.args[0], 0)
        self.load_speech.assert_called_once()
        display.assert_called_once()
        self.text.encode.assert_not_called()
        self.detector.detect.assert_not_called()
        self.sam.segment.assert_not_called()
        self.camera.release.assert_called_once()
        self.close_window.assert_called_once()

    def test_failed_load_leaves_devices_unopened(self):
        self.load_speech.side_effect = FileNotFoundError('missing speech weights')
        with self.assertRaisesRegex(FileNotFoundError, 'missing speech weights'):
            run()
        self.open_camera.assert_not_called()
        self.device.InputStream.assert_not_called()
        self.open_window.assert_not_called()

    def test_playback_continues_during_inference_and_uses_spoken_target(self):
        captured = Queue()
        started, resume, heard, caught_up = Event(), Event(), Event(), Event()
        partial, repeated = Event(), Event()
        stopped = self.enterContext(patch('tinysegment.pipeline.Event', return_value=Event())).return_value
        mask = np.ones((2, 3), dtype=bool)
        self.sam.segment.return_value = {'combined_mask': mask}
        received, processed, shown = [], [], []
        workers = {}

        def capture():
            workers['camera'] = current_thread()
            number = captured.get(timeout=5)
            if number is None:
                self.assertTrue(stopped.wait(5))
                return False, None
            # Use distinct B and R so inference must convert the camera frame.
            frame = np.full((2, 3, 3), (number, 0, 20), dtype=np.uint8)
            return True, frame

        def transcribe(model, chunks, rate, *, on_text):
            workers['speech'] = current_thread()
            self.assertIs(model, self.transcriber)
            for chunk in chunks:
                received.append((chunk.tolist(), rate))
                if len(received) == 1:
                    on_text(1, 'Just talking.')
                    yield 'Just talking.'
                    on_text(2, 'Reachy, buzz')
                    partial.set()
                elif len(received) == 2:
                    on_text(2, 'Reachy, bus.')
                    yield 'Reachy, bus.'
                    heard.set()

        def detect(image, embeddings, prompts, confidence, iou, image_size):
            workers['segment'] = current_thread()
            red, _, number = image.getpixel((0, 0))
            self.assertEqual(red, 20)
            processed.append((number, prompts))
            if number == 1:
                started.set()
                self.assertTrue(resume.wait(5))
            elif number == 3:
                caught_up.set()
            elif number == 4:
                repeated.set()
            return {'boxes': [[0, 0, 2, 1]]}

        def show(frame, overlay, transcript, prompt):
            self.assertIs(current_thread(), main_thread())
            number = int(frame[0, 0, 0])
            shown.append((number, transcript, prompt))
            if number <= 3:
                self.assertIsNone(overlay)
            else:
                self.assertIs(overlay, mask)
            if number == 1:
                self.assertTrue(started.wait(5))
            if number in (1, 2):
                callback = self.device.InputStream.call_args.kwargs['callback']
                sample = 0.25 if number == 1 else 0.5
                callback(np.array([[sample]], np.float32), 1, None, Mock(input_overflow=False))
            if number == 1:
                self.assertTrue(partial.wait(5))
            if number == 2:
                self.assertTrue(heard.wait(5))
            if number == 3:
                resume.set()
                self.assertTrue(caught_up.wait(5))
            if number == 4:
                self.assertTrue(repeated.wait(5))
            captured.put(number + 1 if number < 5 else None)
            return number < 5

        self.camera.read.side_effect = capture
        self.transcribe.side_effect = transcribe
        self.detector.detect.side_effect = detect
        self.show.side_effect = show
        captured.put(1)
        run(initial_prompt=' person ')

        self.assertEqual(self.startup, ['clip', 'detector', 'segmenter', 'speech', 'camera', 'microphone'])
        self.assertEqual(shown, [
            (1, '', 'person'),
            (2, 'Just talking. Reachy, buzz', 'person'),
            (3, 'Just talking. Reachy, bus.', 'person'),
            (4, 'Just talking. Reachy, bus.', 'bus'),
            (5, 'Just talking. Reachy, bus.', 'bus'),
        ])
        self.assertEqual(processed[:3], [(1, ('person',)), (3, ('bus',)), (4, ('bus',))])
        self.assertEqual(self.text.encode.call_args_list, [call(('person',)), call(('bus',))])
        self.assertEqual(received, [([0.25], 48000), ([0.5], 48000)])
        self.assertEqual(len(set(workers.values())), 3)
        self.assertTrue(all(not worker.is_alive() for worker in workers.values()))
        self.camera.release.assert_called_once()
        self.close_window.assert_called_once()
        for resource in (self.mic, self.transcriber):
            resource.__exit__.assert_called_once()

    def test_camera_releases_on_read_failure(self):
        self.camera.read.return_value = False, None
        with self.assertRaisesRegex(RuntimeError, 'Lost video'):
            run()
        self.camera.release.assert_called_once()
        self.mic.__exit__.assert_called_once()
        self.transcriber.__exit__.assert_called_once()
        self.sam.segment.assert_not_called()

    def test_speech_failure_wakes_preview_before_first_frame(self):
        stopped = Event()
        self.enterContext(patch('tinysegment.pipeline.Event', return_value=stopped))

        def read():
            self.assertTrue(stopped.wait(5))
            return False, None

        self.camera.read.side_effect = read
        self.transcribe.side_effect = RuntimeError('transcription failed')
        with self.assertRaisesRegex(RuntimeError, 'transcription failed'):
            run()
        self.show.assert_not_called()
        self.camera.release.assert_called_once()

    def test_inference_failure_reaches_preview(self):
        self.camera.read.return_value = True, np.zeros((2, 3, 3), dtype=np.uint8)
        self.sam.segment.side_effect = RuntimeError('segmentation failed')
        with self.assertRaisesRegex(RuntimeError, 'segmentation failed'):
            run(initial_prompt='person')
        self.camera.release.assert_called_once()
        self.close_window.assert_called_once()

    def test_quit_reports_an_inflight_worker_failure(self):
        started, stopped = Event(), Event()
        self.enterContext(patch('tinysegment.pipeline.Event', return_value=stopped))
        self.camera.read.return_value = True, np.zeros((2, 3, 3), dtype=np.uint8)
        failure = RuntimeError('segmentation failed during shutdown')

        def segment(image, boxes):
            started.set()
            self.assertTrue(stopped.wait(5))
            raise failure

        def quit_preview(frame, mask, transcript, prompt):
            self.assertTrue(started.wait(5))
            return False

        self.sam.segment.side_effect = segment
        self.show.side_effect = quit_preview
        with self.assertRaises(RuntimeError) as raised:
            run(initial_prompt='person')
        self.assertIs(raised.exception, failure)
        self.close_window.assert_called_once()
        self.camera.release.assert_called_once()

    def test_window_mask_is_opaque_and_background_keeps_its_color(self):
        frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        with patch('cv2.imshow') as display, patch('cv2.waitKey', return_value=ord('q')):
            self.assertFalse(show_mask(frame, np.array([[True, False]])))
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(display.call_args.args[1], [[[255, 90, 0], [40, 50, 60]]])

    def test_captions_stay_in_their_corners_and_off_inference_pixels(self):
        frame = np.full((240, 480, 3), 70, dtype=np.uint8)
        for mask in (None, np.ones(frame.shape[:2], bool), np.ones((1, 1), bool)):
            with (
                self.subTest(mask_shape=getattr(mask, 'shape', None)),
                patch('cv2.imshow') as display,
                patch('cv2.waitKey', return_value=ord('q')),
            ):
                show_mask(frame, mask, 'before the command ' * 20 + 'Reachy, my bag.',
                          'my bright red backpack with the silver zipper')
                preview = display.call_args.args[1]
                np.testing.assert_array_equal(frame, np.full_like(frame, 70))
                background = (255, 90, 0) if mask is not None and mask.shape == frame.shape[:2] else 70
                self.assertTrue((preview[:120] == background).all())
                left, right = preview[120:, :240], preview[120:, 240:]
                self.assertTrue(((left.min(axis=2) > 180) & (np.ptp(left, axis=2) == 0)).any())
                self.assertTrue((right == (255, 222, 96)).all(axis=2).any())
                self.assertFalse((left == (255, 222, 96)).all(axis=2).any())
