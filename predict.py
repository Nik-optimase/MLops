#!/usr/bin/env python3

"""
Predict module for the MLops scoring service.

This script loads the trained model and applies it to the processed test
data produced by `preprocess.py`. The outputs are written to the
`/app/output` directory and include:

* `sample_submission.csv` – the primary submission file containing the
  `id` column and the model's predicted probabilities. If a
  `threshold.txt` file is present, it will also output a binary
  prediction column based on the threshold.
* `importances.json` – a JSON mapping of the top‑5 most important
  features to their importance scores (if the underlying model
  exposes feature importances).
* `scores_density.png` – a histogram plot of the predicted scores to
  visualize the distribution of predictions.

The script assumes that the processed data exists at
`/app/work/processed.csv`, which is created by `preprocess.py`.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_threshold() -> float:
    """Load the prediction threshold from threshold.txt if available."""
    thresh_path = Path("threshold.txt")
    if thresh_path.exists():
        try:
            with open(thresh_path, "r", encoding="utf-8") as f:
                value = float(f.read().strip())
                return value
        except Exception:
            pass
    return 0.5


def main() -> None:
    work_path = Path("./work/processed.csv")
    model_path = Path("model.pkl")
    features_path = Path("features.json")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not work_path.exists():
        raise FileNotFoundError(f"Processed data not found at {work_path}")

    # Load data and features
    df = pd.read_csv(work_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    # Extract features (excluding the id)
    X = df[feature_list]
    ids = df["id"].values if "id" in df.columns else np.arange(len(df))

    # Load model
    model = joblib.load(model_path)

    # Get prediction probabilities (assume binary classification and take prob of class 1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # fallback to predict; ensure 1‑D array
        pred = model.predict(X)
        proba = pred.astype(float)

    # Write sample submission (probabilities)
    submission = pd.DataFrame({"id": ids, "target": proba})
    submission.to_csv(output_dir / "sample_submission.csv", index=False)

    # If threshold is provided, write binary predictions
    thresh = load_threshold()
    binary_pred = (proba >= thresh).astype(int)
    submission_bin = pd.DataFrame({"id": ids, "target": binary_pred})
    submission_bin.to_csv(output_dir / "sample_submission_binary.csv", index=False)

    # Save feature importances (top 5) if available
    importances_path = output_dir / "importances.json"
    try:
        fi = None
        # Try scikit‑learn API first
        if hasattr(model, "named_steps"):
            # Attempt to retrieve the final estimator if pipeline
            try:
                final_estimator = model.named_steps.get("dummyclassifier", model.named_steps[next(iter(model.named_steps))])
            except Exception:
                final_estimator = None
        else:
            final_estimator = model
        if final_estimator is not None and hasattr(final_estimator, "feature_importances_"):
            fi = final_estimator.feature_importances_
        if fi is not None:
            series = pd.Series(fi, index=feature_list)
            top5 = series.sort_values(ascending=False).head(5)
            with open(importances_path, "w", encoding="utf-8") as f:
                json.dump({k: float(v) for k, v in top5.items()}, f, ensure_ascii=False, indent=2)
    except Exception:
        # Silently skip if importances cannot be derived
        pass

    # Save density plot of predicted scores
    try:
        plt.figure()
        plt.hist(proba, bins=50, density=True)
        plt.xlabel("Predicted score")
        plt.ylabel("Density")
        plt.title("Distribution of predicted scores")
        plt.tight_layout()
        plt.savefig(output_dir / "scores_density.png", dpi=150)
        plt.close()
    except Exception:
        pass

    print(f"Predictions saved to {output_dir / 'sample_submission.csv'}")


if __name__ == "__main__":
    main()