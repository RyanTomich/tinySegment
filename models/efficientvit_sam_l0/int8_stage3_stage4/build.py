"""Build the two quantized SAM backbone entries from the original checkpoint."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from models.efficientvit_sam_l0.baseline.loader import load_segmenter
from models.storage import sha256, variant_artifacts
from .calibration import calibration_paths


def entry(stage, stage_index=4):
    return stage.model.image_encoder.backbone.stages[stage_index].op_list[0]


def capture(stage, rgb, stage_index=4):
    captured = []
    def save(module, args, output):
        captured.append((args[0].detach(), output.detach()))

    with entry(stage, stage_index).register_forward_hook(save):
        stage.predictor.set_image(np.asarray(rgb))
    (pair,) = captured
    return pair


def operations(block):
    layers = block.main
    return {'expansion': layers.inverted_conv.conv, 'gelu_1': layers.inverted_conv.act,
            'depthwise': layers.depth_conv.conv, 'gelu_2': layers.depth_conv.act,
            'projection': layers.point_conv.conv, 'batch_norm': layers.point_conv.norm}


def folded_block(block):
    ops = deepcopy(operations(block))
    ops['projection'] = nn.utils.fuse_conv_bn_eval(ops['projection'], ops.pop('batch_norm'))
    return nn.Sequential(*ops.values()).eval()


@torch.no_grad()
def build(coco_root):
    from .export import export_block

    torch.set_num_threads(1)
    sam = load_segmenter()
    paths = calibration_paths(coco_root)
    destination = variant_artifacts('int8_stage3_stage4')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the pair only after both exports succeed.
    with tempfile.TemporaryDirectory(prefix='.build-', dir=destination.parent) as temp:
        for index in (3, 4):
            inputs = []
            for path in paths:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    inputs.append(capture(sam, rgb, index)[0])
            block = folded_block(entry(sam, index))
            program = export_block(block, inputs)
            with (Path(temp) / f'stage{index}-int8.pte').open('xb') as stream:
                program.write_to_file(stream)
        try:
            Path(temp).rename(destination)
        except OSError:
            if not all((destination / p.name).is_file() and sha256(destination / p.name) == sha256(p)
                       for p in Path(temp).iterdir()):
                raise FileExistsError(f'Different or incomplete build at {destination}. Move it aside before rebuilding.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, default=Path.home() / '.cache' / 'tinysegment' / 'coco')
    print(build(parser.parse_args().coco_root))
