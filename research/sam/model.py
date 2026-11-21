"""Research EfficientViT-SAM L0 with XNNPACK INT8 backbone entries."""

from torch import nn

from tinysegment import load_segmenter
from tinysegment.weights import sha256
from .checkpoints import RUNS_ROOT


PROGRAM_DIR = RUNS_ROOT / 'research-model'


class RuntimeBlock(nn.Module):
    def __init__(self, path):
        super().__init__()
        from executorch.runtime import Runtime
        self.program = Runtime.get().load_program(path)
        self.method = self.program.load_method('forward')

    def forward(self, x):
        return self.method.execute((x,))[0]


def load_research_segmenter(*, weights_dir=None):
    """Load the CPU research SAM, including both compiled INT8 entry blocks."""
    paths = {index: PROGRAM_DIR / f'stage{index}-int8.pte' for index in (3, 4)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing research SAM program {path}. See research/sam/README.md for setup.')
    stage = load_segmenter(weights_dir=weights_dir, device='cpu')
    for index, path in paths.items():
        stage.model.image_encoder.backbone.stages[index].op_list[0] = RuntimeBlock(path).eval()
        stage.metadata[f'stage{index}_program'] = {'path': str(path), 'sha256': sha256(path)}
    return stage
