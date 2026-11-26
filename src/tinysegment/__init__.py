"""Independent, editable model stages. Heavy dependencies load on demand."""
from models.clip_vit_b32.baseline.loader import load_text_encoder
from models.efficientvit_sam_l0.baseline.loader import load_segmenter
from models.yolo_world_v2_small.baseline.loader import load_detector

__all__ = ['load_text_encoder', 'load_detector', 'load_segmenter']
