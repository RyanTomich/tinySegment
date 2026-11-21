"""Camera capture, mask presentation, and the live inference loop."""
from contextlib import contextmanager

import cv2
from PIL import Image

from .stream import LatestInput


@contextmanager
def camera_frames(index=0):
    """Continuously drain the Mac camera while inference uses the latest frame."""
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    def capture():
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f'Lost video from camera {index}')
            yield frame

    try:
        if not camera.isOpened():
            raise RuntimeError(f'Cannot open camera {index}. Check camera access in macOS System Settings.')
        frames = LatestInput(capture())
        try:
            yield frames
        finally:
            frames.close()
            frames.join()
    finally:
        camera.release()


class MaskWindow:
    title = 'TinySegment'

    def __enter__(self):
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        return self

    def show(self, frame, mask):
        pixels = frame
        if mask is not None and mask.shape == frame.shape[:2]:
            pixels = frame.copy()
            pixels[mask] = (255, 90, 0)
        cv2.imshow(self.title, pixels)
        key = cv2.waitKey(1)
        return key not in (27, ord('q')) and cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) > 0

    def __exit__(self, *exc):
        cv2.destroyAllWindows()


def infer_masks(frames, prompts, pipeline, **options):
    for frame in frames:
        prompt = prompts.current()
        if prompt:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, result = pipeline.process(image, [prompt], **options)
            yield result['combined_mask']


def run_video(frames, masks, output):
    for frame in frames:
        if not output.show(frame, masks.current()):
            break
