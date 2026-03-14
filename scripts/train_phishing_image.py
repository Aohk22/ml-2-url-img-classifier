from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_phishing_image.py` without setting PYTHONPATH.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phishing_image.data import ImageDatasetConfig
from phishing_image.experiment import run_experiment, save_bundle


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a baseline image classifier from folder labels.")
    p.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Dataset root. Expected layout: `{phishing,not-phishing}/<sample>/screenshots/*.jpg` "
            "(also accepts `not-phising/`)."
        ),
    )
    p.add_argument("--image-size", type=int, nargs=2, default=(64, 64), metavar=("W", "H"))
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--glob",
        default="*/screenshots/*",
        help="Path pattern relative to each class folder (e.g. `*/screenshots/*.jpg` or `**/*`).",
    )
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out", default="artifacts/phishing_image.joblib")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = ImageDatasetConfig(
        data_dir=Path(args.data_dir),
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        recursive=bool(args.recursive),
        file_glob=str(args.glob) if args.glob is not None else None,
        max_per_class=args.max_per_class,
        seed=int(args.seed),
    )

    exp, best, candidates = run_experiment(
        cfg,
        random_state=int(args.seed),
        val_size=float(args.val_size),
        test_size=float(args.test_size),
    )

    print("Data dir:", exp.data_dir)
    print("Images:", exp.n_images)
    print("Image size:", exp.image_size)
    print("Best model:", exp.best_model)
    print("VAL metrics:", exp.val_metrics)
    print("TEST metrics:", exp.test_metrics)
    print("All candidates:")
    for c in sorted(candidates, key=lambda r: r.metrics["roc_auc"], reverse=True):
        print(" ", c.name, c.metrics)

    out = save_bundle(args.out, best, exp)
    print("Saved bundle:", out.resolve())


if __name__ == "__main__":
    main()
