"""Compatibility imports for the original baseline stage API."""
from .baseline_models import Stage, device_for
from .baseline_models.clip import TextEncoder, load_text_encoder
from .baseline_models.efficientvit_sam import Segmenter, load_segmenter
from .baseline_models.yolo_world import Detector, configure_ultralytics, load_detector
