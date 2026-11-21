"""Build the two quantized SAM backbone entries from the original checkpoint."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import tempfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
import torch
from torch import nn

from tinysegment import load_segmenter
from tinysegment.pipeline import read_image

# First 16 eligible COCO val2017 images outside the fixed quality evaluation.
CALIBRATION_IDS = (139, 285, 632, 724, 776, 785, 802, 872, 885, 1268, 1296, 1353, 1425, 1490, 1503, 1532)


def calibration_paths(root, ids=CALIBRATION_IDS):
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
    from tinysegment.weights import project_weights_dir

    torch.set_num_threads(1)
    sam = load_segmenter()
    paths = calibration_paths(coco_root)
    runs = project_weights_dir().parent / 'runs' / 'sam'
    runs.mkdir(parents=True, exist_ok=True)
    # Publish the pair only after both exports succeed.
    with tempfile.TemporaryDirectory(dir=runs) as temp:
        for index in (3, 4):
            inputs = [capture(sam, read_image(path), index)[0] for path in paths]
            block = folded_block(entry(sam, index))
            program, _ = export_block('int8', block, inputs[0], inputs)
            with (Path(temp) / f'stage{index}-int8.pte').open('xb') as stream:
                program.write_to_file(stream)
        destination = Path(tempfile.mkdtemp(prefix='int8-stage3-stage4-', dir=runs))
        Path(temp).replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True)
    print(build(parser.parse_args().coco_root))
