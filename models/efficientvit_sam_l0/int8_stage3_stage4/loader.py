"""Research EfficientViT-SAM L0 with XNNPACK INT8 backbone entries."""

from pathlib import Path

from torch import nn

from models.efficientvit_sam_l0.baseline.loader import load_segmenter
from models.storage import variant_artifacts, sha256


def program_paths(program_dir=None, weights_dir=None):
    if program_dir is None:
        program_dir = variant_artifacts('int8_stage3_stage4', weights_dir)
    directory = Path(program_dir).expanduser().resolve()
    return {index: directory / f'stage{index}-int8.pte' for index in (3, 4)}


class RuntimeBlock(nn.Module):
    def __init__(self, path):
        super().__init__()
        try:
            from executorch.runtime import Runtime
        except ImportError as exc:
            raise RuntimeError('Install ExecuTorch to use the int8_stage3_stage4 segmenter') from exc
        self.program = Runtime.get().load_program(path)
        self.method = self.program.load_method('forward')

    def forward(self, x):
        return self.method.execute((x,))[0]


def load_research_segmenter(*, weights_dir=None, checkpoint=None, device='cpu', program_dir=None):
    """Load the CPU research SAM, including both compiled INT8 entry blocks."""
    if str(device) != 'cpu':
        raise ValueError('int8_stage3_stage4 requires the CPU device')
    paths = program_paths(program_dir, weights_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing research SAM program {path}. Build both with python -m models.efficientvit_sam_l0.int8_stage3_stage4.build')
    stage = load_segmenter(weights_dir=weights_dir, checkpoint=checkpoint, device='cpu')
    for index, path in paths.items():
        stage.model.image_encoder.backbone.stages[index].op_list[0] = RuntimeBlock(path).eval()
        stage.metadata[f'stage{index}_program'] = {'path': str(path), 'sha256': sha256(path)}
    return stage
