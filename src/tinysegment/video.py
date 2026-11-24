"""Camera capture, mask presentation, and the live inference loop."""
from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event

import cv2
from PIL import Image

from .stream import LatestInput


@contextmanager
def microphone_prompts(*, weights_dir=None, initial=None):
    """Capture the default microphone continuously while video samples prompts."""
    from .baseline_models.moonshine import load_transcriber
    from .pipeline import latest_speech_prompts

    try:
        import sounddevice as sd
        transcriber = load_transcriber(weights_dir=weights_dir)
    except ImportError as exc:
        raise RuntimeError("Install the video speech dependencies with pip install -e '.[speech]'") from exc

    # Buffer short ASR stalls while keeping the PortAudio callback nonblocking.
    audio = Queue(maxsize=30)
    stopped = Event()
    failure = None

    def capture(samples, frames, timing, status):
        nonlocal failure
        try:
            if not status.input_overflow:
                audio.put_nowait(samples[:, 0].copy())
                return
        except Full:
            pass
        failure = RuntimeError('Microphone input overflowed during transcription')
        raise sd.CallbackAbort

    def chunks(mic):
        while not stopped.is_set():
            if failure:
                raise failure
            try:
                yield audio.get(timeout=0.1)
            except Empty:
                if not mic.active:
                    raise RuntimeError('Microphone capture stopped unexpectedly')

    with transcriber:
        try:
            rate = int(sd.query_devices(kind='input')['default_samplerate'])
            with sd.InputStream(samplerate=rate, channels=1, dtype='float32',
                                blocksize=rate // 10, callback=capture) as mic:
                prompts = latest_speech_prompts(transcriber, chunks(mic), sample_rate=rate, initial=initial)
                try:
                    yield prompts
                finally:
                    stopped.set()
                    prompts.close()
                    prompts.join()
        except sd.PortAudioError as exc:
            raise RuntimeError(f'Cannot use the default microphone. Check the input device and microphone permission. {exc}') from exc


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
