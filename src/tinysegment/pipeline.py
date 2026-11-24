"""Image I/O and composition, independent of tensor-level model execution."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .selection import load_stage
from .stream import LatestInput


def speech_prompts(transcriber, audio_chunks, *, sample_rate=16000):
    """Yield up to ten words after each 'Reachy' in finalized speech.

    A wake word spoken alone arms the next utterance. Pass a preloaded
    transcriber and a continuous source of normalized mono audio chunks.
    """
    from .baseline_models.moonshine import completed_lines

    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    wake_word = re.compile(r'\bReachy\b', re.IGNORECASE)
    armed = False
    for line in completed_lines(transcriber, audio_chunks, sample_rate):
        for index, phrase in enumerate(wake_word.split(line)):
            if index:
                armed = True
            if armed:
                words = re.findall(r"\w+(?:[-'’]\w+)*", phrase)
                if words:
                    armed = False
                    yield ' '.join(words[:10])


def latest_speech_prompts(transcriber, audio_chunks, *, sample_rate=16000, initial=None):
    """Listen in the background and retain the last command in current()."""
    updates = speech_prompts(transcriber, audio_chunks, sample_rate=sample_rate)
    return LatestInput(updates, initial)


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


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
    from .weights import validate_output_path
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        fresh = Path(temporary) / path.name
        image.save(fresh, format='PNG'); fresh.replace(path)


class Pipeline:
    """Resident inference stages for one sequential worker."""

    def __init__(self, *, weights_dir=None, device='cpu', clip_checkpoint=None,
                 detector_checkpoint=None, segmenter_checkpoint=None, clip_variant='baseline',
                 detector_variant='baseline', segmenter_variant='baseline', segmenter_program_dir=None):
        options = dict(weights_dir=weights_dir, device=device)
        self.text = load_stage('clip', clip_variant, checkpoint=clip_checkpoint, **options)
        self.detector = load_stage('detector', detector_variant, checkpoint=detector_checkpoint, **options)
        self.segmenter = load_stage('segmenter', segmenter_variant, checkpoint=segmenter_checkpoint,
                                    program_dir=segmenter_program_dir, **options)
        self._last_prompts = None
        self._embeddings = None

    def process(self, image, prompts, *, legacy_marked_input=False,
                confidence=.25, iou=.7, image_size=640):
        prompts = tuple(prompts)
        if prompts != self._last_prompts:
            self._embeddings = self.text.encode(prompts)
            self._last_prompts = prompts
        detections = self.detector.detect(image, self._embeddings, prompts, confidence, iou, image_size)
        boxes = detections['boxes']
        if legacy_marked_input and boxes:
            image = Image.alpha_composite(image.convert('RGBA'), markers(image, boxes)).convert('RGB')
        return detections, self.segmenter.segment(image, boxes)


def run(image, prompts, *, weights_dir=None, device='cpu', legacy_marked_input=False,
        confidence=.25, iou=.7, image_size=640, clip_checkpoint=None,
        detector_checkpoint=None, segmenter_checkpoint=None, clip_variant='baseline',
        detector_variant='baseline', segmenter_variant='baseline', segmenter_program_dir=None):
    """One-shot compatibility entry point, including the visualization."""
    pipeline = Pipeline(weights_dir=weights_dir, device=device, clip_checkpoint=clip_checkpoint,
                        detector_checkpoint=detector_checkpoint, segmenter_checkpoint=segmenter_checkpoint,
                        clip_variant=clip_variant, detector_variant=detector_variant,
                        segmenter_variant=segmenter_variant, segmenter_program_dir=segmenter_program_dir)
    detections, segmentation = pipeline.process(image, prompts, legacy_marked_input=legacy_marked_input,
                                                confidence=confidence, iou=iou, image_size=image_size)
    return detections, segmentation, render_overlay(image, detections['boxes'], segmentation['combined_mask'])
