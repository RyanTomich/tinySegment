import unittest
from queue import Queue
from threading import Event
from unittest.mock import Mock, patch

import numpy as np

from tinysegment.stream import LatestInput
from tinysegment.video import camera_frames, infer_masks, MaskWindow, run_video


class VideoTests(unittest.TestCase):
    def test_playback_continues_during_inference(self):
        captured = Queue()
        def capture():
            while (frame := captured.get(timeout=5)) is not None:
                yield frame

        frames = LatestInput(capture())
        started = Event()
        release = Event()
        mask = np.ones((2, 3), dtype=bool)
        processed = []
        shown = []
        prompts = Mock()
        prompts.current.return_value = 'person'

        def process(image, labels):
            processed.append((image.getpixel((0, 0))[0], labels))
            started.set()
            if not release.wait(5):
                raise TimeoutError('Playback stalled during inference')
            return {}, {'combined_mask': mask}

        masks = LatestInput(infer_masks(frames, prompts, Mock(process=process)))

        def send(number):
            captured.put(np.full((2, 3, 3), number, np.uint8))

        def show(frame, overlay):
            number = int(frame[0, 0, 0])
            shown.append(number)
            if number <= 3:
                self.assertIsNone(overlay)
            else:
                self.assertIs(overlay, mask)
            if number == 1:
                self.assertTrue(started.wait(5))
            if number == 3:
                prompts.current.return_value = 'bus'
                release.set()
                self.assertIs(next(iter(masks)), mask)
            if number == 5:
                return False
            send(number + 1)
            return True

        try:
            send(1)
            run_video(frames, masks, Mock(show=show))
        finally:
            release.set()
            captured.put(None)
            frames.join()
            masks.join()
        self.assertEqual(shown, [1, 2, 3, 4, 5])
        self.assertEqual(processed[0], (1, ['person']))
        self.assertGreaterEqual(processed[1][0], 3)
        self.assertEqual(processed[1][1], ['bus'])

    def test_camera_releases_on_read_failure(self):
        camera = Mock()
        camera.read.return_value = False, None
        with patch('tinysegment.video.cv2.VideoCapture', return_value=camera):
            with camera_frames() as frames:
                with self.assertRaisesRegex(RuntimeError, 'Lost video'):
                    next(iter(frames))
        camera.release.assert_called_once()

    def test_window_mask_is_opaque_and_background_keeps_its_color(self):
        frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        with patch('tinysegment.video.cv2.imshow') as display, \
             patch('tinysegment.video.cv2.waitKey', return_value=ord('q')):
            self.assertFalse(MaskWindow().show(frame, np.array([[True, False]])))
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(display.call_args.args[1], [[[255, 90, 0], [40, 50, 60]]])
