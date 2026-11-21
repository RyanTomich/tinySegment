"""Research EfficientViT-SAM L0 with XNNPACK INT8 backbone entries."""

from pathlib import Path

from torch import nn

from tinysegment import load_segmenter
from tinysegment.weights import project_weights_dir, sha256


def program_paths(program_dir=None):
    if program_dir is None:
        program_dir = project_weights_dir().parent / 'runs' / 'sam' / 'research-model'
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
    paths = program_paths(program_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing research SAM program {path}. Provide the directory containing both programs with --segmenter-program-dir.')
    stage = load_segmenter(weights_dir=weights_dir, checkpoint=checkpoint, device='cpu')
    for index, path in paths.items():
        stage.model.image_encoder.backbone.stages[index].op_list[0] = RuntimeBlock(path).eval()
        stage.metadata[f'stage{index}_program'] = {'path': str(path), 'sha256': sha256(path)}
    return stage
