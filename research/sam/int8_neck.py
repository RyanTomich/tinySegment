"""Quantize and benchmark the eight middle-neck convolutions of EfficientViT-SAM L0."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.ao import nn as ao_nn
from torch.ao.quantization.observer import MinMaxObserver

from tinysegment import load_segmenter
from tinysegment.models import Segmenter
from tinysegment.weights import sha256, verify_baseline
from .benchmark import SAMPLES, WARMUP, latency, paired_samples
from .checkpoints import allocate_run
from .evaluate import DEFAULT_BOXES, DEFAULT_IMAGE, comparison, read_fixture
from .quality import eligible, MANIFEST, quality

FORMAT = 'efficientvit-sam-l0-int8-neck-v1'


def neck_layers(model):
    blocks = model.image_encoder.neck.middle.op_list
    if len(blocks) != 4:
        raise ValueError('Expected four L0 neck blocks')
    return [layer for block in blocks for layer in (block.main.spatial_conv, block.main.point_conv)]


def quantize_filters(weight):
    peak = weight.abs().amax(dim=(1, 2, 3))
    scales = torch.where(peak > 0, peak / 127, 1.0)
    return torch.quantize_per_channel(weight, scales, torch.zeros_like(scales, dtype=torch.long),
                                      axis=0, dtype=torch.qint8)


def calibration_paths(root):
    annotations = json.loads((root / 'annotations/instances_val2017.json').read_text())
    evaluation_ids = set(map(int, json.loads(MANIFEST.read_text())))
    ids = sorted({ann['image_id'] for ann in annotations['annotations'] if eligible(ann)} - evaluation_ids)[:16]
    if len(ids) != 16:
        raise ValueError('Expected 16 eligible held-out calibration images')
    paths = []
    for image_id in ids:
        path = root / 'val2017' / f'{image_id:012}.jpg'
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
                urlretrieve(f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{path.name}', temp.name)
                with Image.open(temp.name) as image:
                    image.verify()
                try:
                    os.link(temp.name, path)
                except FileExistsError:
                    pass  # Another run published the image first.
        paths.append(path)
    return paths


def observe(observer, tensor):
    observer(tensor)


def calibrate(stage, paths):
    observers = [(MinMaxObserver(), MinMaxObserver()) for _ in neck_layers(stage.model)]
    with ExitStack() as stack:
        for layer, (before, after) in zip(neck_layers(stage.model), observers, strict=True):
            stack.enter_context(layer.conv.register_forward_pre_hook(
                lambda module, args, obs=before: observe(obs, args[0])))
            stack.enter_context(layer.norm.register_forward_hook(
                lambda module, args, output, obs=after: observe(obs, output)))
        for path in paths:
            with Image.open(path) as image:
                stage.predictor.set_image(np.asarray(image.convert('RGB')))
    return observers


def install_neck(model, observers=None):
    for i, layer in enumerate(neck_layers(model)):
        conv = layer.conv
        quant = ao_nn.quantized.Quantize(1.0, 0, torch.quint8)
        kernel = ao_nn.quantized.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                        conv.stride, conv.padding, conv.dilation, conv.groups)
        if observers is not None:
            before, after = observers[i]
            scale, zero = before.calculate_qparams()
            quant = ao_nn.quantized.Quantize(scale.item(), zero.item(), torch.quint8)
            scale, zero = after.calculate_qparams()
            kernel.scale, kernel.zero_point = scale.item(), zero.item()
            folded = nn.utils.fuse_conv_bn_eval(conv, layer.norm)
            kernel.set_weight_bias(quantize_filters(folded.weight.detach()), folded.bias.detach())
        layer.conv = nn.Sequential(quant, kernel, ao_nn.quantized.DeQuantize())
        layer.norm = None


def save_candidate(model, path):
    with path.open('wb') as stream:
        torch.save({'format': FORMAT, 'state': model.state_dict()}, stream)


def load_candidate(path):
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != FORMAT:
        raise ValueError('Unknown research checkpoint format')
    torch.backends.quantized.engine = 'qnnpack'
    model = create_efficientvit_sam_model('efficientvit-sam-l0', pretrained=False).eval()
    install_neck(model)
    model.load_state_dict(saved['state'])
    return Segmenter(model, path)


def time_neck(stages):
    middle = [stage.model.image_encoder.neck.middle for stage in stages]
    rgb, _ = read_fixture(DEFAULT_IMAGE, DEFAULT_BOXES)
    inputs = []
    with middle[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0])):
        stages[0].predictor.set_image(np.asarray(rgb))
    return paired_samples([partial(block, inputs[0]) for block in middle])


@torch.inference_mode()
def experiment(root, directory, report):
    original = load_segmenter()
    paths = calibration_paths(root)
    report['calibration_images'] = [path.name for path in paths]
    candidate = load_segmenter()
    observers = calibrate(candidate, paths)
    install_neck(candidate.model, observers)
    path = directory.write('checkpoint.pt', partial(save_candidate, candidate.model))
    del candidate
    candidate = load_candidate(path)
    stages = [original, candidate]
    report.update(original_sha256=verify_baseline('segmenter', original.checkpoint),
                  candidate_sha256=sha256(path), checkpoint_bytes=comparison(
                      original.checkpoint.stat().st_size, path.stat().st_size))
    print('Scoring held-out COCO masks', flush=True)
    report['quality'] = quality(*stages, root)
    directory.json('quality.json', report['quality'])
    print('Timing FP32 no-op control', flush=True)
    control = load_segmenter()
    report['noop'] = latency([original, control])
    del control
    report['neck'] = time_neck(stages)
    print('Timing full SAM', flush=True)
    report['latency'] = [latency(stages)]
    verify_baseline('segmenter', original.checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.quantized.engine = 'qnnpack'
    torch.set_num_threads(1)
    directory = allocate_run('int8-neck')
    print(directory.path, flush=True)
    report = {'format': FORMAT, 'engine': 'qnnpack', 'warmup': WARMUP, 'samples': SAMPLES}
    try:
        experiment(args.coco_root, directory, report)
        report['status'] = 'success'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        directory.json('report.json', report)


if __name__ == '__main__':
    main()
