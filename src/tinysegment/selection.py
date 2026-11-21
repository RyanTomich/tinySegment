"""Resolve a stage and variant once, before processing images."""
from .baseline_models.clip import load_text_encoder
from .baseline_models.efficientvit_sam import load_segmenter
from .baseline_models.yolo_world import load_detector


def _load_int8(**options):
    from .experimental_models.efficientvit_sam.int8_stage3_stage4 import load_research_segmenter
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
