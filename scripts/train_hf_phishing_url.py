from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/train_hf_phishing_url.py` without setting PYTHONPATH.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hf_phishing_url.experiment import load_default_splits, run_experiment, save_bundle


def main() -> None:
    splits = load_default_splits()
    exp, best, candidates = run_experiment(splits, random_state=42, val_size=0.2)

    print("Dataset:", exp.dataset_id)
    print("Features:", len(exp.feature_cols))
    print("Best model:", exp.best_model)
    print("VAL metrics:", exp.val_metrics)
    print("TEST metrics:", exp.test_metrics)
    print("All candidates:")
    for c in sorted(candidates, key=lambda r: r.metrics["roc_auc"], reverse=True):
        print(" ", c.name, c.metrics)

    out = save_bundle(Path("artifacts") / "hf_pirocheto_phishing_url.joblib", best, exp)
    print("Saved bundle:", out.resolve())


if __name__ == "__main__":
    main()
