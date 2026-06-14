"""Phase 2: train & validate the KAN on the derived features.

Honest, reproducible pipeline:
  - stratified train / val / test split (test is never seen during training)
  - standardize features (scaler params saved for inference)
  - train a small KAN
  - report REAL accuracy overall AND on the hard/ambiguous subset
  - save model checkpoint, scaler, metrics.json, confusion matrix + loss curve

Run:  python -m model.train
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from kan import KAN

from core.config import SEED, Intent, INTENT_NAMES
from core.features import FEATURE_NAMES

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.parquet")
MODEL_DIR = os.path.dirname(__file__)
CKPT_PATH = os.path.join(MODEL_DIR, "kan_ckpt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npz")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

torch.manual_seed(SEED)
np.random.seed(SEED)
CLASS_NAMES = [INTENT_NAMES[i] for i in Intent]


def load_split():
    """Load features and make a stratified 70/15/15 train/val/test split."""
    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["intent"].to_numpy(dtype=np.int64)
    hard = df["hard"].to_numpy(dtype=np.int64)

    # 70 / 15 / 15, stratified by class. Carry the hard flag through the split.
    X_tr, X_tmp, y_tr, y_tmp, h_tr, h_tmp = train_test_split(
        X, y, hard, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te, h_val, h_te = train_test_split(
        X_tmp, y_tmp, h_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return (X_tr, y_tr, h_tr), (X_val, y_val, h_val), (X_te, y_te, h_te)


def standardize(X_tr, X_val, X_te):
    """Standardize using TRAIN statistics only; save params for inference."""
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    np.savez(SCALER_PATH, mean=mean, std=std, feature_names=np.array(FEATURE_NAMES))
    return [(X - mean) / std for X in (X_tr, X_val, X_te)]


def main():
    (X_tr, y_tr, h_tr), (X_val, y_val, h_val), (X_te, y_te, h_te) = load_split()
    X_tr, X_val, X_te = standardize(X_tr, X_val, X_te)
    print(f"Split: train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}  | features={len(FEATURE_NAMES)}")

    dataset = {
        "train_input": torch.tensor(X_tr, dtype=torch.float32),
        "train_label": torch.tensor(y_tr, dtype=torch.long),
        "test_input": torch.tensor(X_val, dtype=torch.float32),   # val used for monitoring
        "test_label": torch.tensor(y_val, dtype=torch.long),
    }

    # Small KAN: 18 inputs -> 8 hidden -> 3 classes. Parameter-efficient by design.
    model = KAN(width=[len(FEATURE_NAMES), 8, len(Intent)], grid=3, k=3, seed=SEED)
    loss_fn = torch.nn.CrossEntropyLoss()

    def train_acc():
        pred = model(dataset["train_input"]).argmax(dim=1)
        return (pred == dataset["train_label"]).float().mean()

    def val_acc():
        pred = model(dataset["test_input"]).argmax(dim=1)
        return (pred == dataset["test_label"]).float().mean()

    # Train in short chunks and keep the checkpoint with the best VALIDATION loss
    # (early stopping). lamb adds L1 regularization on KAN activations to curb the
    # train/val divergence (overfitting) seen with unregularized LBFGS.
    print("Training KAN...")
    STEPS_PER_CHUNK, MAX_CHUNKS, PATIENCE = 3, 14, 3
    results = {"train_loss": [], "test_loss": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    for _ in range(MAX_CHUNKS):
        r = model.fit(dataset, opt="LBFGS", steps=STEPS_PER_CHUNK, loss_fn=loss_fn,
                      lamb=0.002, metrics=(train_acc, val_acc))
        results["train_loss"] += list(r["train_loss"])
        results["test_loss"] += list(r["test_loss"])
        with torch.no_grad():
            v = float(loss_fn(model(dataset["test_input"]), dataset["test_label"]))
        if v < best_val - 1e-4:
            best_val, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:                 # restore best-on-validation weights
        model.load_state_dict(best_state)
    print(f"Early-stopped; best val loss = {best_val:.4f}")

    # --- honest evaluation on the held-out TEST set --------------------------
    with torch.no_grad():
        logits_te = model(torch.tensor(X_te, dtype=torch.float32))
        pred_te = logits_te.argmax(dim=1).numpy()

    overall_acc = float((pred_te == y_te).mean())
    easy_mask, hard_mask = (h_te == 0), (h_te == 1)
    easy_acc = float((pred_te[easy_mask] == y_te[easy_mask]).mean())
    hard_acc = float((pred_te[hard_mask] == y_te[hard_mask]).mean())
    cm = confusion_matrix(y_te, pred_te)

    print(f"\n=== TEST ({len(y_te)} held-out tracks) ===")
    print(f"Overall accuracy : {overall_acc:.4f}")
    print(f"  easy subset    : {easy_acc:.4f}  ({easy_mask.sum()} tracks)")
    print(f"  hard subset    : {hard_acc:.4f}  ({hard_mask.sum()} tracks)")
    print("\nClassification report:")
    print(classification_report(y_te, pred_te, target_names=CLASS_NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # --- feature attribution (KAN's "why") -----------------------------------
    feature_importance = {}
    try:
        scores = model.feature_score.detach().numpy()
        feature_importance = {FEATURE_NAMES[i]: float(scores[i]) for i in range(len(FEATURE_NAMES))}
    except Exception:
        try:
            model.attribute()
            scores = model.feature_score.detach().numpy()
            feature_importance = {FEATURE_NAMES[i]: float(scores[i]) for i in range(len(FEATURE_NAMES))}
        except Exception as e:
            print(f"(feature attribution unavailable: {e})")

    # --- save artifacts ------------------------------------------------------
    model.saveckpt(CKPT_PATH)
    metrics = {
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "overall_accuracy": overall_acc,
        "easy_accuracy": easy_acc,
        "hard_accuracy": hard_acc,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "feature_importance": feature_importance,
        "architecture": {"width": [len(FEATURE_NAMES), 8, len(Intent)], "grid": 3, "k": 3},
        "seed": SEED,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model -> {CKPT_PATH}*  | scaler -> {SCALER_PATH}  | metrics -> {METRICS_PATH}")

    # --- plots ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if "train_loss" in results:
        ax[0].plot(results["train_loss"], label="train")
    if "test_loss" in results:
        ax[0].plot(results["test_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("step"); ax[0].set_yscale("log"); ax[0].legend()
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax[1], colorbar=False)
    ax[1].set_title(f"Confusion matrix (test, acc={overall_acc:.3f})")
    fig.tight_layout()
    out = os.path.join(MODEL_DIR, "training_report.png")
    fig.savefig(out, dpi=110)
    print(f"Saved plot  -> {out}")

    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:6]
        print("\nTop features (KAN attribution):")
        for name, sc in top:
            print(f"  {name:32s}: {sc:.3f}")


if __name__ == "__main__":
    main()
