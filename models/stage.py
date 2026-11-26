"""Shared helpers for resident model stages."""
from models.storage import MANIFEST


def device_for(value):
    import torch
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f'CUDA device unavailable: {value}')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS device unavailable')
    if device.type not in {'cpu', 'cuda', 'mps'}:
        raise ValueError(f'Unsupported device: {value}; choose cpu, cuda[:index], or mps')
    return device


class Stage:
    def __init__(self, model, path, name):
        self.model = model.float().eval()
        self.checkpoint = path
        self.metadata = {'component': name, 'checkpoint': str(path), 'baseline': MANIFEST[name]}
