import pickle

from .. import Stage, device_for
from ...weights import checkpoint_path


class TextEncoder(Stage):
    def tokenize(self, prompts):
        import clip
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError('Provide at least one nonempty text prompt')
        return clip.tokenize([p.strip() for p in prompts], truncate=True).to(next(self.model.parameters()).device)

    def forward(self, tokens):
        """Differentiable tensor path; normalization matches YOLO-World's CLIP adapter."""
        features = self.model.encode_text(tokens).float()
        return (features / features.norm(dim=-1, keepdim=True)).unsqueeze(0)

    def encode(self, prompts):
        import torch
        with torch.inference_mode():
            return self.forward(self.tokenize(prompts))


def load_text_encoder(*, weights_dir=None, checkpoint=None, device='cpu'):
    import clip
    path = checkpoint_path('clip', weights_dir, checkpoint)
    try:
        model, preprocess = clip.load(str(path), device=device_for(device), jit=False)
    except (RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Incompatible CLIP checkpoint {path}: {exc}') from exc
    stage = TextEncoder(model, path, 'clip')
    stage.image_preprocess = preprocess
    return stage
