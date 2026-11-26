import pickle

from models.stage import Stage, device_for
from models.storage import checkpoint_path


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


def load_segmenter(*, weights_dir=None, checkpoint=None, device='cpu'):
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    path = checkpoint_path('segmenter', weights_dir, checkpoint)
    target = device_for(device)
    try:
        model = create_efficientvit_sam_model('efficientvit-sam-l0', weight_url=str(path)).to(target)
    except (RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Incompatible EfficientViT-SAM checkpoint {path}: {exc}') from exc
    return Segmenter(model, path)
