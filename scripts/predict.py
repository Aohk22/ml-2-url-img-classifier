from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/predict.py` without setting PYTHONPATH.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with models in `models/`.")
    sub = p.add_subparsers(dest="cmd", required=True)

    url_p = sub.add_parser("url", help="Predict phishing probability from URL(s).")
    url_p.add_argument(
        "--model",
        default="models/url_clf_features_only.joblib",
        help="Path to a joblib-saved sklearn model (default: features-only URL model).",
    )
    url_p.add_argument("--threshold", type=float, default=0.5)
    url_p.add_argument("--json", action="store_true", help="Output JSON lines.")
    url_p.add_argument("urls", nargs="+", help="One or more URLs.")

    img_p = sub.add_parser("image", help="Predict phishing probability from a screenshot image.")
    img_p.add_argument("--weights", default="models/img_clf_model.pt", help="Path to `.pt` state_dict.")
    img_p.add_argument("--threshold", type=float, default=0.5)
    img_p.add_argument("--image-size", type=int, default=224)
    img_p.add_argument("--device", default=None, help="Torch device (e.g. cpu, cuda).")
    img_p.add_argument("--json", action="store_true", help="Output JSON.")
    img_p.add_argument("image_path", help="Path to an image file.")

    return p.parse_args()


def _cmd_url(args: argparse.Namespace) -> int:
    from hf_phishing_url.inference import load_url_pipeline, predict_urls

    pipe = load_url_pipeline(args.model)
    preds = predict_urls(pipe, args.urls, threshold=float(args.threshold))

    if args.json:
        for p in preds:
            print(json.dumps(p.__dict__, ensure_ascii=False))
        return 0

    for p in preds:
        print(f"{p.phishing_proba:.4f}\t{int(p.is_phishing)}\t{p.url}")
    return 0


def _cmd_image(args: argparse.Namespace) -> int:
    from phishing_image.inference import load_image_model, predict_image

    model = load_image_model(
        args.weights,
        device=args.device,
        image_size=int(args.image_size),
    )
    pred = predict_image(
        model,
        args.image_path,
        threshold=float(args.threshold),
        device=args.device,
        image_size=int(args.image_size),
    )

    if args.json:
        print(json.dumps(pred.__dict__, ensure_ascii=False))
        return 0

    print(f"{pred.phishing_proba:.4f}\t{int(pred.is_phishing)}\t{pred.image_path}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.cmd == "url":
        return _cmd_url(args)
    if args.cmd == "image":
        return _cmd_image(args)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
