"""Box-prompted COCO accuracy and agreement with the original segmenter."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tinysegment.weights import sha256

MANIFEST = Path(__file__).with_name('coco_val2017_100.json')


def eligible(ann):
    return (not ann.get('iscrowd', 0) and ann['area'] > 0 and ann['segmentation']
            and all(np.isfinite(ann['bbox'])) and min(ann['bbox'][2:]) > 0)


def mask_iou(left, right):
    if left.shape != right.shape:
        raise ValueError('Mask dimensions differ')
    union = np.count_nonzero(left | right)
    return np.count_nonzero(left & right) / union if union else 1.0


def quality(original, candidate, root, full=False):
    from pycocotools.coco import COCO

    annotations = Path(root) / 'annotations/instances_val2017.json'
    coco = COCO(str(annotations))
    manifest = None if full else json.loads(MANIFEST.read_text())
    image_ids = sorted(coco.imgs) if full else [int(key) for key in manifest]
    totals = {key: np.zeros(3) for key in ('all', 'small', 'medium', 'large')}
    counts = dict.fromkeys(totals, 0)
    images = 0
    for image_id in image_ids:
        info = coco.imgs[image_id]
        anns = sorted((a for a in coco.imgToAnns[image_id] if eligible(a)), key=lambda a: a['id'])
        if manifest is not None and [a['id'] for a in anns] != manifest[str(image_id)]:
            raise ValueError(f'COCO annotations differ from manifest for image {image_id}')
        if not anns:
            continue
        with Image.open(Path(root) / 'val2017' / info['file_name']) as source:
            rgb = source.convert('RGB')  # Annotation coordinates use the stored raster.
        if rgb.size != (info['width'], info['height']):
            raise ValueError(f'Unexpected dimensions for image {image_id}')
        pixels = np.asarray(rgb)
        for stage in (original, candidate):
            stage.predictor.set_image(pixels)
        images += 1
        for ann in anns:
            x, y, w, h = ann['bbox']
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            truth = coco.annToMask(ann).astype(bool)
            masks = []
            for stage in (original, candidate):
                mask, score, _ = stage.predictor.predict(box=box, multimask_output=False)
                if mask.shape != (1, *truth.shape) or not np.isfinite(score).all():
                    raise ValueError(f"Invalid prediction for annotation {ann['id']}")
                masks.append(mask[0])
            scores = [mask_iou(masks[0], truth), mask_iou(masks[1], truth), mask_iou(*masks)]
            area = ann['area']
            group = 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'
            for key in ('all', group):
                counts[key] += 1
                totals[key] += scores
    if not images:
        raise ValueError('No eligible COCO instances')
    metrics = {}
    for key, total in totals.items():
        count = counts[key]
        row = {'instances': count}
        if count:
            baseline, experiment, fidelity = 100 * total / count
            row.update(original_miou_pct=baseline, candidate_miou_pct=experiment,
                       delta_points=experiment - baseline, fidelity_miou_pct=fidelity)
        metrics[key] = row
    return {'split': 'val2017', 'images': images, 'annotations_sha256': sha256(annotations),
            'manifest_sha256': None if full else sha256(MANIFEST), 'groups': metrics}
