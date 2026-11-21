import argparse
import inspect
import json
import sys
from pathlib import Path

from .selection import VARIANTS, load_stage
from .stream import LatestInput, latest_prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent editable text, detection, and segmentation models')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['encode-text', 'detect', 'segment', 'run', 'video', 'inspect', 'setup-weights']:
        p = sub.add_parser(name)
        p.add_argument('--weights-dir', type=Path)
        if name == 'setup-weights': continue
        p.add_argument('--device', default='cpu')
        if name not in ('run', 'video'): p.add_argument('--checkpoint', type=Path)
        if name in ['detect', 'segment', 'run']: p.add_argument('image', type=Path)
        if name == 'encode-text': p.add_argument('prompts', nargs='+')
        if name == 'run':
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument('prompt', nargs='?')
            source.add_argument('--stream', action='store_true',
                                help='Read the latest prompt from stdin and write a binary mask for each result')
            p.add_argument('--legacy-marked-input', action='store_true')
        if name == 'video':
            p.add_argument('prompt', nargs='?', help='Initial prompt for the live camera')
            p.add_argument('--camera', type=int, default=0)
        if name in ('run', 'video'):
            for stage, variants in VARIANTS.items():
                p.add_argument(f'--{stage}-checkpoint', type=Path)
                p.add_argument(f'--{stage}-variant', choices=variants, default='baseline')
        else:
            p.add_argument('--variant', default='baseline')
        if name in ('run', 'video', 'segment', 'inspect'):
            p.add_argument('--segmenter-program-dir', type=Path)
        if name == 'detect': p.add_argument('--embeddings', type=Path, required=True)
        if name == 'segment': p.add_argument('--boxes', type=Path, required=True)
        if name in ['detect', 'run', 'video']:
            p.add_argument('--confidence', type=float, default=.25)
            p.add_argument('--iou', type=float, default=.7)
            p.add_argument('--image-size', type=int, default=640)
        if name == 'inspect': p.add_argument('model', choices=['clip','detector','segmenter'])
        elif name != 'video': p.add_argument('--output', type=Path, default=Path('mask.png') if name == 'run' else None, required=name != 'run')
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


def execute(args):
    from .weights import setup_weights, sha256, validate_output_path
    if args.command == 'setup-weights':
        setup_weights(args.weights_dir); return
    options = dict(weights_dir=args.weights_dir, device=args.device)
    if args.command not in ('run', 'video'):
        options.update(checkpoint=args.checkpoint, variant=args.variant,
                       program_dir=getattr(args, 'segmenter_program_dir', None))
    if args.command == 'inspect':
        stage = load_stage(args.model, **options)
        print(stage.model)
        print(json.dumps({**stage.metadata, 'sha256':sha256(stage.checkpoint), 'parameters':sum(p.numel() for p in stage.model.parameters()), 'source':inspect.getfile(type(stage.model))}, indent=2))
        return
    import numpy as np
    from .pipeline import Pipeline, read_image, render_overlay, save_png
    if args.command in ('run', 'video'):
        model_options = dict(options, clip_checkpoint=args.clip_checkpoint,
                             detector_checkpoint=args.detector_checkpoint, segmenter_checkpoint=args.segmenter_checkpoint,
                             clip_variant=args.clip_variant, detector_variant=args.detector_variant,
                             segmenter_variant=args.segmenter_variant, segmenter_program_dir=args.segmenter_program_dir)
        inference_options = dict(confidence=args.confidence, iou=args.iou, image_size=args.image_size)
    if args.command == 'video':
        from .video import camera_frames, infer_masks, MaskWindow, run_video
        pipeline = Pipeline(**model_options)
        with camera_frames(args.camera) as frames:
            print('Enter a prompt in the terminal. Press q in the video window to quit.', flush=True)
            prompts = latest_prompts(sys.stdin, initial=(args.prompt or '').strip())
            masks = LatestInput(infer_masks(frames, prompts, pipeline, **inference_options))
            try:
                with MaskWindow() as output:
                    run_video(frames, masks, output)
            finally:
                frames.close()
                prompts.close()
                masks.close()
                masks.join()
        return
    inputs = [getattr(args, attribute, None) for attribute in
              ['image', 'embeddings', 'boxes', 'checkpoint', 'clip_checkpoint',
               'detector_checkpoint', 'segmenter_checkpoint']]
    if (args.command == 'segment' and args.variant == 'int8_stage3_stage4'
            or args.command == 'run' and args.segmenter_variant == 'int8_stage3_stage4'):
        from .experimental_models.efficientvit_sam.int8_stage3_stage4 import program_paths
        inputs.extend(program_paths(args.segmenter_program_dir).values())
    args.output = validate_output_path(args.output, weights_dir=args.weights_dir, inputs=inputs)
    if args.command == 'encode-text':
        labels = [p.strip() for p in args.prompts]
        features = load_stage('clip', **options).encode(labels).cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as stream: np.savez_compressed(stream, embeddings=features, labels=np.asarray(labels))
    else:
        image = read_image(args.image)
        if args.command == 'detect':
            with np.load(args.embeddings, allow_pickle=False) as data:
                detections = load_stage('detector', **options).detect(image, data['embeddings'], data['labels'].tolist(), args.confidence, args.iou, args.image_size)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(detections, indent=2)+'\n')
        elif args.command == 'segment':
            data = json.loads(args.boxes.read_text())
            if 'image_size' in data and data['image_size'] != list(image.size):
                raise ValueError('Boxes image_size does not match the input image')
            result = load_stage('segmenter', **options).segment(image, data['boxes'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('wb') as stream: np.savez_compressed(stream, **result)
        else:
            from PIL import Image
            if not args.stream and not args.prompt.strip():
                raise ValueError('Text prompt must not be empty')
            pipeline = Pipeline(**model_options)
            prompts = latest_prompts(sys.stdin) if args.stream else [args.prompt.strip()]
            for prompt in prompts:
                detections, result = pipeline.process(
                    image, [prompt], legacy_marked_input=args.legacy_marked_input, **inference_options)
                mask = result['combined_mask']
                output = Image.fromarray(mask) if args.stream else render_overlay(image, detections['boxes'], mask)
                save_png(output, args.output)
                print(f"{len(detections['boxes'])} detection(s). Saved {args.output}", flush=True)
            return
    print(f'Saved {args.output}')
