import argparse
import inspect
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent editable text, detection, and segmentation models')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['encode-text', 'detect', 'segment', 'run', 'inspect', 'setup-weights']:
        p = sub.add_parser(name)
        p.add_argument('--weights-dir', type=Path)
        if name == 'setup-weights': continue
        p.add_argument('--device', default='cpu')
        if name != 'run': p.add_argument('--checkpoint', type=Path)
        if name in ['detect', 'segment', 'run']: p.add_argument('image', type=Path)
        if name == 'encode-text': p.add_argument('prompts', nargs='+')
        if name == 'run':
            p.add_argument('prompt')
            p.add_argument('--legacy-marked-input', action='store_true')
            for model in ['clip', 'detector', 'segmenter']: p.add_argument(f'--{model}-checkpoint', type=Path)
        if name == 'detect': p.add_argument('--embeddings', type=Path, required=True)
        if name == 'segment': p.add_argument('--boxes', type=Path, required=True)
        if name in ['detect', 'run']:
            p.add_argument('--confidence', type=float, default=.25)
            p.add_argument('--iou', type=float, default=.7)
            p.add_argument('--image-size', type=int, default=640)
        if name == 'inspect': p.add_argument('model', choices=['clip','detector','segmenter'])
        else: p.add_argument('--output', type=Path, default=Path('mask.png') if name == 'run' else None, required=name != 'run')
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


def execute(args):
    from .weights import setup_weights, sha256, validate_output_path
    if args.command == 'setup-weights':
        setup_weights(args.weights_dir); return
    from .models import load_text_encoder, load_detector, load_segmenter
    loaders = {'clip':load_text_encoder, 'detector':load_detector, 'segmenter':load_segmenter}
    options = dict(weights_dir=args.weights_dir, device=args.device)
    if args.command != 'run': options['checkpoint'] = args.checkpoint
    if args.command == 'inspect':
        stage = loaders[args.model](**options)
        print(stage.model)
        print(json.dumps({**stage.metadata, 'sha256':sha256(stage.checkpoint), 'parameters':sum(p.numel() for p in stage.model.parameters()), 'source':inspect.getfile(type(stage.model))}, indent=2))
        return
    import numpy as np
    from .pipeline import read_image, run, save_png
    inputs = [getattr(args, attribute, None) for attribute in
              ['image', 'embeddings', 'boxes', 'checkpoint', 'clip_checkpoint',
               'detector_checkpoint', 'segmenter_checkpoint']]
    args.output = validate_output_path(args.output, weights_dir=args.weights_dir, inputs=inputs)
    if args.command == 'encode-text':
        labels = [p.strip() for p in args.prompts]
        features = load_text_encoder(**options).encode(labels).cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as stream: np.savez_compressed(stream, embeddings=features, labels=np.asarray(labels))
    else:
        image = read_image(args.image)
        if args.command == 'detect':
            with np.load(args.embeddings, allow_pickle=False) as data:
                detections = load_detector(**options).detect(image, data['embeddings'], data['labels'].tolist(), args.confidence, args.iou, args.image_size)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(detections, indent=2)+'\n')
        elif args.command == 'segment':
            data = json.loads(args.boxes.read_text())
            if 'image_size' in data and data['image_size'] != list(image.size):
                raise ValueError('Boxes image_size does not match the input image')
            result = load_segmenter(**options).segment(image, data['boxes'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('wb') as stream: np.savez_compressed(stream, **result)
        else:
            if not args.prompt.strip(): raise ValueError('Text prompt must not be empty')
            detections, _, overlay = run(image, [args.prompt.strip()], **options,
                legacy_marked_input=args.legacy_marked_input, confidence=args.confidence,
                iou=args.iou, image_size=args.image_size, clip_checkpoint=args.clip_checkpoint,
                detector_checkpoint=args.detector_checkpoint, segmenter_checkpoint=args.segmenter_checkpoint)
            save_png(overlay, args.output)
            print(f"{len(detections['boxes'])} detection(s)")
    print(f'Saved {args.output}')
