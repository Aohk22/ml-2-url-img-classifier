"""Image-based phishing classifier utilities."""

from .inference import ImagePrediction, load_image_model, predict_image
from .model import SimpleCNN

__all__ = ["ImagePrediction", "SimpleCNN", "load_image_model", "predict_image"]
