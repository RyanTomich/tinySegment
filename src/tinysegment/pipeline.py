"""Model loading, sequential image inference, and the live camera session."""
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
import re
from tempfile import TemporaryDirectory
from threading import Condition, Event, Thread

import numpy as np
from PIL import Image, ImageDraw

from models.clip_vit_b32.baseline.loader import load_text_encoder
from models.efficientvit_sam_l0.baseline.loader import load_segmenter
from models.storage import validate_output_path
from models.yolo_world_v2_small.baseline.loader import load_detector
from tinysegment.captions import draw_captions


_WINDOW_TITLE = 'TinySegment'


def _load_int8(**options):
    from models.efficientvit_sam_l0.int8_stage3_stage4.loader import load_research_segmenter
    return load_research_segmenter(**options)


VARIANTS = {
    'clip': {'baseline': load_text_encoder},
    'detector': {'baseline': load_detector},
    'segmenter': {'baseline': load_segmenter, 'int8_stage3_stage4': _load_int8},
}


def load_stage(stage, variant='baseline', *, weights_dir=None, checkpoint=None,
               device='cpu', program_dir=None):
    variants = VARIANTS.get(stage, {})
    if variant not in variants:
        raise ValueError(f'Unknown variant {variant!r} for {stage}. Available: {", ".join(variants)}')
    options = dict(weights_dir=weights_dir, checkpoint=checkpoint, device=device)
    if program_dir is not None:
        if (stage, variant) != ('segmenter', 'int8_stage3_stage4'):
            raise ValueError('A program directory requires the int8_stage3_stage4 segmenter')
        options['program_dir'] = program_dir
    return variants[variant](**options)


def load_models(*, weights_dir=None, device='cpu', clip_checkpoint=None, detector_checkpoint=None,
                segmenter_checkpoint=None, clip_variant='baseline', detector_variant='baseline',
                segmenter_variant='baseline', segmenter_program_dir=None):
    options = dict(weights_dir=weights_dir, device=device)
    text = load_stage('clip', clip_variant, checkpoint=clip_checkpoint, **options)
    detector = load_stage('detector', detector_variant, checkpoint=detector_checkpoint, **options)
    segmenter = load_stage('segmenter', segmenter_variant, checkpoint=segmenter_checkpoint,
                           program_dir=segmenter_program_dir, **options)
    return text, detector, segmenter


def segment_image(image, prompts, embeddings, detector, segmenter, *, legacy_marked_input=False,
                  confidence=.25, iou=.7, image_size=640):
    """Detect prompted objects and segment their boxes in detection order."""
    detections = detector.detect(image, embeddings, prompts, confidence, iou, image_size)
    boxes = detections['boxes']
    if legacy_marked_input and boxes:
        image = Image.alpha_composite(image.convert('RGBA'), markers(image, boxes)).convert('RGB')
    return detections, segmenter.segment(image, boxes)


def run(camera_index=0, initial_prompt=None, *, weights_dir=None, confidence=.25,
        iou=.7, image_size=640, **model_options):
    """Run the voice-controlled live camera session with preview on the main thread."""
    import cv2
    from models.moonshine_tiny_streaming_en.baseline.loader import load_transcriber

    text, detector, segmenter = load_models(weights_dir=weights_dir, **model_options)
    try:
        import sounddevice as sd
        transcriber = load_transcriber(weights_dir=weights_dir)
    except ImportError as exc:
        raise RuntimeError("Install the video speech dependencies with pip install -e '.[speech]'") from exc

    stopped = Event()
    latest_frame = _BroadcastSlot()
    requested_prompt = _BroadcastSlot((initial_prompt or '').strip())
    active_prompt = _BroadcastSlot(requested_prompt.current())
    latest_transcript = _BroadcastSlot('')
    latest_mask = _BroadcastSlot()
    worker_errors = SimpleQueue()

    with (
        transcriber,
        open_camera(camera_index) as camera,
        microphone_chunks(sd, stopped) as (audio, rate),
    ):
        def capture():
            while not stopped.is_set():
                ok, frame = camera.read()
                if stopped.is_set():
                    return
                if not ok:
                    raise RuntimeError(f'Lost video from camera {camera_index}')
                latest_frame.publish(frame)

        def listen():
            recent_lines = {}

            def update_caption(line_id, text):
                # Moonshine revises the whole line as it recognizes more speech.
                recent_lines[line_id] = text
                if len(recent_lines) > 3:
                    del recent_lines[next(iter(recent_lines))]
                latest_transcript.publish(' '.join(recent_lines.values()).strip())

            for prompt in speech_prompts(transcriber, audio, sample_rate=rate, on_text=update_caption):
                requested_prompt.publish(prompt)

        def segment():
            last_prompt = None
            for frame in latest_frame:
                if stopped.is_set():
                    break
                prompt = requested_prompt.current()
                if not prompt:
                    continue
                labels = (prompt,)
                if prompt != last_prompt:
                    active_prompt.publish(prompt)
                    embedding = text.encode(labels)
                    last_prompt = prompt
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                _, result = segment_image(image, labels, embedding, detector, segmenter,
                                          confidence=confidence, iou=iou, image_size=image_size)
                latest_mask.publish(result['combined_mask'])

        def run_worker(task):
            try:
                task()
            except Exception as exc:
                worker_errors.put(exc)
                stopped.set()
                latest_frame.close()

        workers = [Thread(name=task.__name__, target=run_worker, args=(task,))
                   for task in (listen, capture, segment)]
        cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        try:
            for worker in workers:
                worker.start()
            print('Say "Reachy" followed by your target. Press q in the video window to quit.', flush=True)
            for frame in latest_frame:
                if stopped.is_set() or not show_mask(frame, latest_mask.current(),
                                                     latest_transcript.current(), active_prompt.current()):
                    break
        finally:
            stopped.set()
            latest_frame.close()
            for worker in workers:
                if worker.ident is not None:
                    worker.join()
            cv2.destroyAllWindows()
    if not worker_errors.empty():
        raise worker_errors.get()


class _BroadcastSlot:
    """A single slot. Each iterator sees the newest update since its own last read."""

    def __init__(self, initial=None):
        self._changed = Condition()
        self._value = initial
        self._version = 0
        self._closed = False
        self._error = None

    def publish(self, value):
        with self._changed:
            if not self._closed:
                self._value = value
                self._version += 1
                self._changed.notify_all()

    def current(self):
        with self._changed:
            return self._value

    def close(self, error=None):
        with self._changed:
            if self._error is None:
                self._error = error
            self._closed = True
            self._changed.notify_all()

    def __iter__(self):
        version = 0
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._version != version or self._closed)
                if self._version == version:
                    if self._error is not None:
                        raise self._error
                    return
                value, version = self._value, self._version
            yield value


def latest_prompts(lines):
    """Read stdin independently so only the latest pending prompt reaches inference."""
    prompts = _BroadcastSlot()

    def read():
        try:
            for line in lines:
                prompt = line.strip()
                if prompt:
                    prompts.publish(prompt)
        except Exception as exc:
            prompts.close(exc)
        else:
            prompts.close()

    Thread(target=read, name='stdin', daemon=True).start()
    return prompts


def speech_prompts(transcriber, audio_chunks, *, sample_rate=16000, on_text=None):
    """Yield up to ten words after each 'Reachy' in finalized speech.

    A wake word spoken alone arms the next utterance. Pass a preloaded
    transcriber and a continuous source of normalized mono audio chunks.
    """
    from models.moonshine_tiny_streaming_en.baseline.loader import completed_lines

    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    wake_word = re.compile(r'\bReachy\b', re.IGNORECASE)
    armed = False
    for line in completed_lines(transcriber, audio_chunks, sample_rate, on_text=on_text):
        for index, phrase in enumerate(wake_word.split(line)):
            if index:
                armed = True
            if armed:
                words = re.findall(r"\w+(?:[-'’]\w+)*", phrase)
                if words:
                    armed = False
                    yield ' '.join(words[:10])


@contextmanager
def microphone_chunks(sd, stopped):
    """Keep the audio callback nonblocking and deliver consecutive mono chunks to speech."""
    audio = Queue(maxsize=30)
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

    try:
        rate = int(sd.query_devices(kind='input')['default_samplerate'])
        with sd.InputStream(samplerate=rate, channels=1, dtype='float32',
                            blocksize=rate // 10, callback=capture) as mic:
            yield chunks(mic), rate
    except sd.PortAudioError as exc:
        raise RuntimeError(f'Cannot use the default microphone. Check the input device and microphone permission. {exc}') from exc


@contextmanager
def open_camera(index):
    import cv2
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Cannot open camera {index}. Check camera access in macOS System Settings.')
        yield camera
    finally:
        camera.release()


def show_mask(frame, mask, transcript='', prompt=''):
    """Composite the mask and captions on a preview copy of the shared camera frame."""
    import cv2
    pixels = frame.copy()
    if mask is not None and mask.shape == frame.shape[:2]:
        pixels[mask] = (255, 90, 0)
    draw_captions(pixels, transcript, prompt)
    cv2.imshow(_WINDOW_TITLE, pixels)
    key = cv2.waitKey(1)
    return key not in (27, ord('q')) and cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) > 0


def markers(image, boxes):
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    for x1, y1, x2, y2 in boxes:
        x, y = round((x1 + x2) / 2), round((y1 + y2) / 2)
        draw.line((x-5, y, x+5, y), fill=(0,235,65,255), width=3)
        draw.line((x, y-5, x, y+5), fill=(0,235,65,255), width=3)
    return layer


def render_overlay(image, boxes, combined_mask):
    if not len(boxes):
        return image.convert('RGBA')
    faded = image.convert('RGBA'); faded.putalpha(90)
    blue = Image.new('RGBA', image.size, (0,90,255))
    blue.putalpha(Image.fromarray(combined_mask.astype(np.uint8) * 128))
    output = Image.alpha_composite(faded, blue)
    layer = Image.new('RGBA', image.size); draw = ImageDraw.Draw(layer)
    for box in np.clip(boxes, 0, (image.width-1, image.height-1)*2):
        draw.rectangle(tuple(round(v) for v in box), outline=(0,235,65,85), width=2)
    return Image.alpha_composite(Image.alpha_composite(output, layer), markers(image, boxes))


def save_png(image, path):
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        fresh = Path(temporary) / path.name
        image.save(fresh, format='PNG'); fresh.replace(path)


if __name__ == '__main__':
    run()
