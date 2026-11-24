"""
Compare the original VGG checkpoint (vgg.pt) and the Optuna VGG checkpoint
(vgg_optuna.pt) on the EXACT SAME patient-wise validation samples.

Run from C:\\Users\\jfd10\\DL_Term:
python code\\compare_vgg_models.py --csv data\\T2_data\\labels.csv --split-csv task2_split.csv

The script:
- Does not train or modify either checkpoint.
- Uses each checkpoint's saved preprocessing metadata when available.
- For legacy checkpoints without resize_mode metadata, assumes torchvision
  Resize((H, W)) (the original vgg.py preprocessing).
- Reports BA, sensitivity, specificity, precision, accuracy and ROC-AUC.
- Reports metrics at 0.5, each checkpoint's saved threshold, and a
  validation-optimal threshold (clearly marked as optimistic/descriptive).
- Saves per-image probabilities and a metrics CSV in vgg_comparison/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "T2_data" / "labels.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "task2_split.csv"
DEFAULT_BASELINE = PROJECT_ROOT / "vgg.pt"
DEFAULT_OPTUNA = PROJECT_ROOT / "vgg_optuna.pt"
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path.name} does not contain 'model_state_dict'. "
            "This comparison script expects the project VGG checkpoint format."
        )
    return ckpt


def build_model(dropout: float) -> nn.Module:
    # Same VGG11 architecture used in the project; no pretrained weights.
    model = models.vgg11(weights=None, init_weights=False)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(float(dropout)),
        nn.Linear(256, 1),
    )
    return model


def resolve_image_path(value: str, csv_path: Path) -> Path:
    raw = str(value).strip()
    normalized = raw.replace("\\", "/")
    rel = Path(normalized)
    candidates = []

    # Absolute path if valid on this machine.
    if rel.is_absolute():
        candidates.append(rel)

    # Common paths in this project.
    candidates.extend([
        csv_path.parent / rel,
        csv_path.parent / "images" / rel.name,
        csv_path.parent.parent / rel,
        PROJECT_ROOT / rel,
        PROJECT_ROOT / "data" / rel,
        PROJECT_ROOT / "data" / "T2_data" / "images" / rel.name,
    ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find image '{raw}'. Checked paths relative to {csv_path.parent}. "
        "Check the 'path' column in labels.csv."
    )


class ResizeWithPadding:
    def __init__(self, size, fill=(0, 0, 0)):
        self.target_h, self.target_w = int(size[0]), int(size[1])
        self.fill = fill

    def __call__(self, image):
        src_w, src_h = image.size
        scale = min(self.target_w / src_w, self.target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize(
            (new_w, new_h), resample=Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (self.target_w, self.target_h), self.fill)
        left = (self.target_w - new_w) // 2
        top = (self.target_h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


def make_eval_transform(ckpt):
    image_size = tuple(ckpt.get("image_size", [160, 112]))
    resize_mode = ckpt.get("resize_mode")
    # Legacy vgg.pt checkpoints from the original script did not store
    # resize_mode; that script used transforms.Resize((H, W)).
    if resize_mode in (None, "direct_resize", "resize"):
        resize = transforms.Resize(image_size)
        mode_used = "direct_resize (legacy default)" if resize_mode is None else "direct_resize"
    elif resize_mode in ("aspect_preserving_padding", "resize_with_padding"):
        resize = ResizeWithPadding(image_size)
        mode_used = "aspect_preserving_padding"
    else:
        raise ValueError(
            f"Unsupported resize_mode in checkpoint: {resize_mode!r}. "
            "Please tell me the metadata shown by the script."
        )
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform, image_size, mode_used


class ValidationDataset(Dataset):
    def __init__(self, frame, csv_path, transform):
        self.paths = [resolve_image_path(x, csv_path) for x in frame["path"]]
        self.labels = frame["label"].astype(int).to_numpy()
        self.sample_ids = frame["sample_id"].astype(str).to_numpy()
        self.patient_groups = frame["patient_group"].astype(str).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = self.transform(im.convert("RGB"))
        return image, int(self.labels[index])


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs = []
    for images, _ in loader:
        logits = model(images.to(device)).view(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def get_threshold_metrics(y, probs, threshold):
    pred = (probs >= float(threshold)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_validation_threshold(y, probs):
    # Descriptive only: this optimizes a threshold on the same validation labels.
    candidates = np.linspace(0.10, 0.90, 161)
    scores = [balanced_accuracy_score(y, (probs >= t).astype(int)) for t in candidates]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--optuna", default=str(DEFAULT_OPTUNA))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "vgg_comparison"))
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    split_path = Path(args.split_csv).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    optuna_path = Path(args.optuna).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {csv_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"task2_split.csv not found: {split_path}")

    df = pd.read_csv(csv_path, dtype={"sample_id": str, "patient_group": str})
    required = {"sample_id", "path", "label", "patient_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("labels.csv contains missing required values.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["patient_group"] = df["patient_group"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in labels.csv.")

    split_df = pd.read_csv(split_path, dtype={"sample_id": str})
    if not {"sample_id", "split"}.issubset(split_df.columns):
        raise ValueError("task2_split.csv must contain sample_id and split columns.")
    if split_df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in task2_split.csv.")
    if set(split_df["sample_id"].astype(str)) != set(df["sample_id"].astype(str)):
        raise ValueError("task2_split.csv sample IDs do not exactly match labels.csv.")
    split_map = dict(zip(split_df["sample_id"].astype(str), split_df["split"].astype(str)))
    split_values = df["sample_id"].map(split_map)
    if not set(split_values.unique()).issubset({"train", "valid"}):
        raise ValueError("Split column must contain only 'train' and 'valid'.")
    valid_df = df.loc[split_values.to_numpy() == "valid"].reset_index(drop=True)
    train_df = df.loc[split_values.to_numpy() == "train"].reset_index(drop=True)

    train_groups = set(train_df["patient_group"])
    valid_groups = set(valid_df["patient_group"])
    if train_groups & valid_groups:
        raise ValueError("Patient leakage detected: same patient in train and validation.")
    if valid_df["label"].nunique() != 2:
        raise ValueError("Validation split must contain both classes.")

    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    baseline_ckpt = load_checkpoint(baseline_path)
    optuna_ckpt = load_checkpoint(optuna_path)

    # If a checkpoint records its split hash, verify it matches the local split.
    split_hash_verified = {}
    for name, ckpt in [("baseline", baseline_ckpt), ("Optuna", optuna_ckpt)]:
        expected = ckpt.get("split_sha256") or ckpt.get("official_split_sha256")
        if expected:
            if expected != split_hash:
                raise ValueError(
                    f"{name} checkpoint was trained with a different task2_split.csv "
                    f"(checkpoint hash {expected[:12]}..., current hash {split_hash[:12]}...). "
                    "Do not compare on this split until the correct split file is restored."
                )
            split_hash_verified[name] = True
        else:
            split_hash_verified[name] = False

    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    print("=== VGG CHECKPOINT COMPARISON ===")
    print(f"Device: {device}")
    print(f"Train: {len(train_df)} images / {len(train_groups)} patients")
    print(f"Validation: {len(valid_df)} images / {len(valid_groups)} patients")
    print(f"Validation labels: {valid_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Split SHA256: {split_hash}")
    if not split_hash_verified["baseline"]:
        print("WARNING: baseline vgg.pt has no saved split hash; its original split cannot be proven from the checkpoint.")
    if split_hash_verified["Optuna"]:
        print("Optuna checkpoint split hash: verified against this task2_split.csv")
    print()

    all_metrics = []
    prediction_frame = valid_df[["sample_id", "patient_group", "label"]].copy()
    y = valid_df["label"].to_numpy(dtype=int)

    for name, checkpoint, checkpoint_path in [
        ("Baseline_VGG", baseline_ckpt, baseline_path),
        ("Optuna_VGG", optuna_ckpt, optuna_path),
    ]:
        params = checkpoint.get("optuna_best_params", {})
        dropout = float(params.get("dropout", 0.3))
        model = build_model(dropout)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        transform, image_size, resize_mode = make_eval_transform(checkpoint)
        dataset = ValidationDataset(valid_df, csv_path, transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        probabilities = predict(model.to(device), loader, device)
        prediction_frame[f"{name}_probability"] = probabilities

        saved_threshold = float(checkpoint.get("threshold", 0.5))
        val_best_threshold = best_validation_threshold(y, probabilities)
        auc = float(roc_auc_score(y, probabilities))

        print(f"--- {name} ---")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Preprocessing: {resize_mode} | image_size(H,W)={image_size}")
        print(f"Saved threshold: {saved_threshold:.3f}")
        print(f"ROC-AUC: {auc:.4f}")

        threshold_rows = [
            ("fixed_0.500", 0.5),
            ("checkpoint_saved_threshold", saved_threshold),
            ("validation_optimal_threshold_descriptive_only", val_best_threshold),
        ]
        seen = set()
        for label, threshold in threshold_rows:
            if round(float(threshold), 8) in seen:
                continue
            seen.add(round(float(threshold), 8))
            metrics = get_threshold_metrics(y, probabilities, threshold)
            metrics.update({
                "model": name,
                "threshold_type": label,
                "roc_auc": auc,
                "checkpoint": checkpoint_path.name,
                "resize_mode": resize_mode,
                "image_height": int(image_size[0]),
                "image_width": int(image_size[1]),
            })
            all_metrics.append(metrics)
            print(
                f"{label}: threshold={metrics['threshold']:.3f}"
                f" | BA={metrics['balanced_accuracy']:.4f}"
                f" | Sens={metrics['sensitivity']:.4f}"
                f" | Spec={metrics['specificity']:.4f}"
                f" | Precision={metrics['precision']:.4f}"
                f" | CM=[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]"
            )
        print()
        del model, dataset, loader

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / "vgg_comparison_metrics.csv"
    preds_path = output_dir / "vgg_comparison_predictions.csv"
    metadata_path = output_dir / "vgg_comparison_summary.json"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(preds_path, index=False, encoding="utf-8-sig")

    summary = {
        "validation_images": int(len(valid_df)),
        "validation_patients": int(len(valid_groups)),
        "validation_class_counts": {
            str(k): int(v) for k, v in valid_df["label"].value_counts().sort_index().items()
        },
        "split_sha256": split_hash,
        "checkpoint_split_hash_verified": split_hash_verified,
        "metrics_file": str(metrics_path),
        "predictions_file": str(preds_path),
        "warning": (
            "Validation-optimal threshold metrics are optimistic because the threshold "
            "is selected and evaluated on the same validation labels. Prefer fixed 0.5, "
            "checkpoint threshold with caution, and ROC-AUC for threshold-independent ranking."
        ),
    }
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FILES SAVED ===")
    print(metrics_path)
    print(preds_path)
    print(metadata_path)
    print("No model was trained or modified. Both original checkpoints were only loaded for inference.")


if __name__ == "__main__":
    main()
