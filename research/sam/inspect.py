"""Inspect architecture and parameter layout without dumping tensor values."""
import argparse
from tinysegment import load_segmenter
from tinysegment.weights import sha256


def describe(stage):
    model = stage.model
    lines = [
        'EfficientViT-SAM L0', f'Checkpoint: {stage.checkpoint}',
        f'SHA-256: {sha256(stage.checkpoint)}',
        f'Total parameters: {sum(p.numel() for p in model.parameters()):,}',
        '\nParameters by component:',
    ]
    for name, module in model.named_children():
        lines.append(f'  {name}: {sum(p.numel() for p in module.parameters()):,}')
    lines.extend(['\nArchitecture:', str(model), '\nParameters (name | shape | dtype | trainable):'])
    for name, parameter in model.named_parameters():
        lines.append(f'{name} | {list(parameter.shape)} | {parameter.dtype} | {parameter.requires_grad}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--weights-dir')
    args = parser.parse_args()
    try:
        print(describe(load_segmenter(checkpoint=args.checkpoint, weights_dir=args.weights_dir)), end='')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
