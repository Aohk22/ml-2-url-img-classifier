from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Image model inference requires PyTorch. Install it (e.g. `pip install torch torchvision`)."
        ) from e
    return torch, nn


def SimpleCNN(*, num_classes: int = 2, image_size: int = 224):  # noqa: N802
    """
    Simple 2-block CNN from `notebooks/train-image-model.ipynb`.

    The saved weights in `models/img_clf_model.pt` are a `state_dict()` for this model.
    """
    torch, nn = _require_torch()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * (image_size // 4) * (image_size // 4), 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):  # noqa: ANN001
            x = self.conv(x)
            x = self.fc(x)
            return x

    return _Model()
