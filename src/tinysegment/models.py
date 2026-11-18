"""Thin stage adapters; .model always exposes the editable eager PyTorch module."""
from pathlib import Path
import os
import pickle
from .weights import checkpoint_path, MANIFEST


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


def configure_ultralytics():
    # Keep upstream runtime settings inside a writable, user-local cache.
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path.home() / '.cache' / 'tinysegment' / 'ultralytics'))
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


class Stage:
    def __init__(self, model, path, name):
        self.model = model.float().eval()
        self.checkpoint = path
        self.metadata = {'component': name, 'checkpoint': str(path), 'baseline': MANIFEST[name]}


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


class Detector(Stage):
    def preprocess(self, image, image_size=640):
        import cv2
        import numpy as np
        import torch
        if image_size <= 0 or image_size % 32:
            raise ValueError('image_size must be a positive multiple of 32')
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        ratio = min(image_size / h, image_size / w)
        nw, nh = round(w * ratio), round(h * ratio)
        dw, dh = (image_size - nw) % 32 / 2, (image_size - nh) % 32 / 2
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else rgb
        padded = cv2.copyMakeBorder(resized, round(dh - .1), round(dh + .1), round(dw - .1), round(dw + .1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
        return tensor.to(next(self.model.parameters()).device).float() / 255

    def forward(self, images, embeddings):
        self.model.model[-1].nc = embeddings.shape[1]
        return self.model.predict(images, txt_feats=embeddings)

    def postprocess(self, prediction, input_shape, image_size, count, confidence=.25, iou=.7):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes
        result = non_max_suppression(prediction, conf_thres=confidence, iou_thres=iou, nc=count, max_det=300)[0]
        result[:, :4] = scale_boxes(input_shape, result[:, :4], (image_size[1], image_size[0]))
        return {'boxes': result[:, :4].cpu().tolist(), 'scores': result[:, 4].cpu().tolist(), 'class_ids': result[:, 5].int().cpu().tolist()}

    def detect(self, image, embeddings, labels, confidence=.25, iou=.7, image_size=640):
        import torch
        if not labels or any(not isinstance(s, str) or not s.strip() for s in labels):
            raise ValueError('Embedding labels must be nonempty strings')
        features = torch.as_tensor(embeddings, dtype=torch.float32, device=next(self.model.parameters()).device)
        if features.shape != (1, len(labels), 512) or not torch.isfinite(features).all():
            raise ValueError('Embeddings must be finite with shape [1, number_of_labels, 512]')
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError('confidence and iou must be between 0 and 1')
        self.model.names = dict(enumerate(labels))
        with torch.inference_mode():
            tensor = self.preprocess(image, image_size)
            result = self.postprocess(self.forward(tensor, features), tensor.shape[2:], image.size, len(labels), confidence, iou)
        return {**result, 'labels': list(labels), 'image_size': list(image.size)}


class Segmenter(Stage):
    def __init__(self, model, path):
        super().__init__(model, path, 'segmenter')
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        self.predictor = EfficientViTSamPredictor(self.model)

    def segment(self, image, boxes):
        import numpy as np
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError('Boxes must be finite [N, 4] xyxy coordinates')
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Each box must have x2 > x1 and y2 > y1')
        masks, scores = [], []
        if len(boxes):
            self.predictor.set_image(np.asarray(image.convert('RGB')))
            for box in boxes:
                mask, score, _ = self.predictor.predict(box=box, multimask_output=False)
                masks.append(mask[0]); scores.append(score[0])
        masks = np.asarray(masks, dtype=bool).reshape(len(boxes), image.height, image.width)
        return {'masks': masks, 'scores': np.asarray(scores, dtype=np.float32), 'combined_mask': masks.any(axis=0)}


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


def load_detector(*, weights_dir=None, checkpoint=None, device='cpu'):
    import torch
    configure_ultralytics()
    path = checkpoint_path('detector', weights_dir, checkpoint)
    target = device_for(device)
    # The upstream checkpoint contains a pickled module: use only trusted checkpoints.
    # Direct loading avoids high-level predictor fusion and hidden CLIP construction.
    try:
        checkpoint_data = torch.load(path, map_location='cpu', weights_only=False)
        model = checkpoint_data.get('ema') or checkpoint_data['model']
        from ultralytics.nn.tasks import WorldModel
        if not isinstance(model, WorldModel):
            raise ValueError('Expected a YOLO-World WorldModel')
        model.clip_model = None
        model = model.to(target)
    except (RuntimeError, KeyError, EOFError, AttributeError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Incompatible YOLO-World checkpoint {path}: {exc}') from exc
    return Detector(model, path, 'detector')


def load_segmenter(*, weights_dir=None, checkpoint=None, device='cpu'):
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    path = checkpoint_path('segmenter', weights_dir, checkpoint)
    target = device_for(device)
    try:
        model = create_efficientvit_sam_model('efficientvit-sam-l0', weight_url=str(path)).to(target)
    except (RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Incompatible EfficientViT-SAM checkpoint {path}: {exc}') from exc
    return Segmenter(model, path)
