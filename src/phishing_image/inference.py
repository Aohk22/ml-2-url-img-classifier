from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ImagePrediction:
    image_path: str
    phishing_proba: float
    is_phishing: bool
    predicted_label: Literal["not-phishing", "phishing"]


def load_image_model(
    weights_path: str | Path,
    *,
    device: str | None = None,
    image_size: int = 224,
) -> Any:
    """Load the CNN weights saved by `notebooks/train-image-model.ipynb`."""
    import torch  # type: ignore

    from .model import SimpleCNN

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN(num_classes=2, image_size=int(image_size))
    state = torch.load(Path(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: Any,
    image_path: str | Path,
    *,
    threshold: float = 0.5,
    device: str | None = None,
    image_size: int = 224,
) -> ImagePrediction:
    """Predict phishing probability for a single screenshot image."""
    import torch  # type: ignore
    from PIL import Image

    if device is None:
        device = next(model.parameters()).device.type

    img = Image.open(Path(image_path)).convert("RGB").resize((int(image_size), int(image_size)))
    x = _pil_to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        phishing_proba = float(probs[1])

    is_phishing = phishing_proba >= float(threshold)
    return ImagePrediction(
        image_path=str(image_path),
        phishing_proba=phishing_proba,
        is_phishing=is_phishing,
        predicted_label="phishing" if is_phishing else "not-phishing",
    )


def _pil_to_tensor(img):  # noqa: ANN001
    """
    Convert PIL image to float32 torch tensor in [0, 1] with shape (3, H, W).

    Implemented without torchvision to keep dependencies minimal.
    """
    import numpy as np
    import torch  # type: ignore

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)
