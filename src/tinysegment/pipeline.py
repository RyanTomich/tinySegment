"""Image I/O and composition, independent of tensor-level model execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .models import load_text_encoder, load_detector, load_segmenter


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


def run(image, prompts, *, weights_dir=None, device='cpu', legacy_marked_input=False,
        confidence=.25, iou=.7, image_size=640, clip_checkpoint=None,
        detector_checkpoint=None, segmenter_checkpoint=None):
    text = load_text_encoder(weights_dir=weights_dir, checkpoint=clip_checkpoint, device=device)
    embeddings = text.encode(prompts)
    detector = load_detector(weights_dir=weights_dir, checkpoint=detector_checkpoint, device=device)
    detections = detector.detect(image, embeddings, prompts, confidence, iou, image_size)
    boxes = detections['boxes']
    if boxes:
        sam_image = Image.alpha_composite(image.convert('RGBA'), markers(image, boxes)).convert('RGB') if legacy_marked_input else image
        segmenter = load_segmenter(weights_dir=weights_dir, checkpoint=segmenter_checkpoint, device=device)
        segmentation = segmenter.segment(sam_image, boxes)
    else:
        segmentation = {'masks': np.zeros((0,image.height,image.width), dtype=bool), 'scores': np.zeros(0,dtype=np.float32), 'combined_mask': np.zeros((image.height,image.width),dtype=bool)}
    return detections, segmentation, render_overlay(image, boxes, segmentation['combined_mask'])
