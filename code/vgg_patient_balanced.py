"""
Task 2 - VGG11 patient-balanced training candidate.

This is a controlled improvement over the supplied vgg.py:
- Keeps the same VGG11 architecture family and no-pretrained-weights rule.
- Keeps the same base evaluation preprocessing: Resize((160,112)),
  ToTensor(), Normalize((0.5,)*3, (0.5,)*3).
- Keeps the existing task2_split.csv patient-wise split; it never creates
  a new split if the required shared split file is missing.
- Uses patient/class-balanced sampling in TRAIN only.
- Adds mild brightness/contrast augmentation in TRAIN only.
- Selects the checkpoint by macro patient ROC-AUC (not by repeatedly
  optimizing a threshold on the same validation set).
- Saves to vgg_patientbalanced.pt; does not overwrite vgg.pt or vgg_optuna.pt.

Run from project root:
python code/vgg_patient_balanced.py --csv data/T2_data/labels.csv \
  --split-csv task2_split.csv --output vgg_patientbalanced.pt --device cuda

No pretrained weights, external data, ensemble, or other feature extractor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "T2_data" / "labels.csv"
DEFAULT_SPLIT = ROOT / "task2_split.csv"
DEFAULT_OUTPUT = ROOT / "vgg_patientbalanced.pt"

SEED = 42
IMAGE_SIZE = (160, 112)  # Keep the existing shared input size (H, W).
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sample_id": str, "patient_group": str})
    required = {"sample_id", "path", "label", "patient_group"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"labels.csv missing columns: {missing}")
    if df[list(required)].isna().any().any():
        raise ValueError("labels.csv contains missing required values.")
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in labels.csv.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["patient_group"] = df["patient_group"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1.")
    return df.reset_index(drop=True)


def resolve_image_path(value: str, csv_path: Path) -> Path:
    raw = str(value).strip().replace("\\", "/")
    relative = Path(raw)
    candidates = []
    if relative.is_absolute():
        candidates.append(relative)
    candidates.extend([
        csv_path.parent / relative,
        csv_path.parent / "images" / relative,
        csv_path.parent / "images" / relative.name,
        csv_path.parent.parent / relative,
        ROOT / relative,
        ROOT / "data" / relative,
        ROOT / "data" / "T2_data" / "images" / relative.name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image not found for path={value!r}; checked relative to {csv_path.parent}"
    )


def load_shared_split(df: pd.DataFrame, split_path: Path):
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Required shared split file not found: {split_path}. "
            "Do not generate a different split for this model."
        )
    split_df = pd.read_csv(split_path, dtype={"sample_id": str})
    if not {"sample_id", "split"}.issubset(split_df.columns):
        raise ValueError("task2_split.csv must contain sample_id and split.")
    if split_df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in task2_split.csv.")
    if set(split_df["sample_id"].astype(str)) != set(df["sample_id"].astype(str)):
        raise ValueError("task2_split.csv IDs do not exactly match labels.csv.")
    mapping = dict(zip(split_df["sample_id"].astype(str), split_df["split"].astype(str)))
    split_values = df["sample_id"].map(mapping)
    if split_values.isna().any() or not set(split_values.unique()).issubset({"train", "valid"}):
        raise ValueError("Split values must be exactly 'train' and 'valid'.")

    train_idx = np.flatnonzero(split_values.to_numpy() == "train")
    valid_idx = np.flatnonzero(split_values.to_numpy() == "valid")
    train_groups = set(df.iloc[train_idx]["patient_group"])
    valid_groups = set(df.iloc[valid_idx]["patient_group"])
    if train_groups & valid_groups:
        raise ValueError(f"Patient leakage detected: {sorted(train_groups & valid_groups)}")
    if df.iloc[train_idx]["label"].nunique() != 2:
        raise ValueError("Training split must contain both labels.")
    if df.iloc[valid_idx]["label"].nunique() != 2:
        raise ValueError("Validation split must contain both labels.")
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    return train_idx, valid_idx, split_hash


def build_transforms(training: bool, brightness_contrast: float):
    operations = [transforms.Resize(IMAGE_SIZE)]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        if brightness_contrast > 0:
            operations.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=brightness_contrast,
                        contrast=brightness_contrast,
                    )],
                    p=0.8,
                )
            )
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(operations)


class UltrasoundDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, csv_path: Path,
                 training: bool, brightness_contrast: float):
        self.paths = [
            resolve_image_path(p, csv_path) for p in frame["path"].astype(str)
        ]
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.sample_ids = frame["sample_id"].astype(str).to_numpy()
        self.patient_groups = frame["patient_group"].astype(str).to_numpy()
        self.transform = build_transforms(training, brightness_contrast)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image_file:
            image = self.transform(image_file.convert("RGB"))
        return image, torch.tensor(self.labels[index], dtype=torch.float32)


def patient_class_weights(frame: pd.DataFrame) -> torch.DoubleTensor:
    """
    Equal total sampling mass per patient; within each patient, divide that
    patient's mass equally among the labels it actually has. Within each
    patient-label cell, sample each image uniformly.

    This prevents patients with more images / more positive frames from
    dominating training. Patient IDs are used only to construct TRAIN
    sampling weights and are never model inputs.
    """
    patient_label_counts = frame.groupby(
        ["patient_group", "label"], observed=True
    ).size()
    labels_per_patient = frame.groupby(
        "patient_group", observed=True
    )["label"].nunique()
    n_patients = frame["patient_group"].nunique()

    weights = []
    for patient, label in zip(frame["patient_group"], frame["label"]):
        cell_count = int(patient_label_counts.loc[(patient, label)])
        n_labels = int(labels_per_patient.loc[patient])
        weight = 1.0 / (n_patients * n_labels * cell_count)
        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid patient/class sampling weights.")
    return torch.as_tensor(weights, dtype=torch.double)


def build_model(dropout: float) -> nn.Module:
    # Same VGG11 architecture and tensor dimensions as the supplied vgg.py.
    # No ImageNet / pretrained weights.
    model = models.vgg11(weights=None, init_weights=False)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(float(dropout)),
        nn.Linear(256, 1),
    )
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)
    return model


@torch.inference_mode()
def predict_probabilities(model, loader, device, use_amp: bool) -> np.ndarray:
    model.eval()
    probabilities = []
    for images, _ in loader:
        if device.type == "cuda":
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            images = images.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images).view(-1)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probabilities)


def safe_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def patient_macro_auc(frame: pd.DataFrame, probabilities: np.ndarray):
    temp = frame[["patient_group", "label"]].copy()
    temp["probability"] = probabilities
    scores = []
    per_patient = {}
    for patient, group in temp.groupby("patient_group", sort=True):
        auc = safe_auc(group["label"].to_numpy(), group["probability"].to_numpy())
        per_patient[str(patient)] = auc
        if np.isfinite(auc):
            scores.append(auc)
    macro = float(np.mean(scores)) if scores else float("nan")
    return macro, per_patient


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def descriptive_best_threshold(y_true, probabilities):
    # This is reported only after the model checkpoint has been selected.
    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        probabilities,
    ]))
    scored = [
        metrics_at_threshold(y_true, probabilities, threshold)["balanced_accuracy"]
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scored))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--brightness-contrast", type=float, default=0.12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    split_path = Path(args.split_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"sample_id": str, "patient_group": str})
    required = {"sample_id", "path", "label", "patient_group"}
    if not required.issubset(df.columns):
        raise ValueError(f"labels.csv must contain {sorted(required)}")
    if df[list(required)].isna().any().any():
        raise ValueError("labels.csv has missing required values.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["patient_group"] = df["patient_group"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")

    train_idx, valid_idx, split_hash = load_shared_split(df, split_path)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    seed_everything(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = UltrasoundDataset(
        train_df, csv_path, training=True,
        brightness_contrast=args.brightness_contrast,
    )
    valid_dataset = UltrasoundDataset(
        valid_df, csv_path, training=False,
        brightness_contrast=0.0,
    )

    sampler = WeightedRandomSampler(
        weights=patient_class_weights(train_df),
        num_samples=len(train_dataset),
        replacement=True,
    )

    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": use_amp,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, **common_loader_args,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size * 2,
        shuffle=False, **common_loader_args,
    )

    model = build_model(args.dropout).to(device)
    if use_amp:
        model = model.to(memory_format=torch.channels_last)

    # Patient/class-balanced sampling already balances the training draw;
    # do not apply a second large positive-class weight.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    y_valid = valid_df["label"].to_numpy(dtype=int)
    best_score = -float("inf")
    best_epoch = 0
    wait = 0
    best_state = None

    train_patient_counts = train_df.groupby("patient_group").size().to_dict()
    train_class_counts = (
        train_df.groupby(["patient_group", "label"]).size().to_dict()
    )

    print("=== VGG11 PATIENT-BALANCED TRAINING ===", flush=True)
    print(f"Device: {device} | AMP: {use_amp}", flush=True)
    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Train: {len(train_df)} images / {train_df.patient_group.nunique()} patients", flush=True)
    print(f"Valid: {len(valid_df)} images / {valid_df.patient_group.nunique()} patients", flush=True)
    print(f"Validation labels: {valid_df.label.value_counts().sort_index().to_dict()}", flush=True)
    print(f"Split SHA256: {split_hash}", flush=True)
    print("Patient-class-balanced sampler: ON", flush=True)
    print(f"Train patient image counts: {train_patient_counts}", flush=True)
    print(f"Train patient/class counts: {train_class_counts}", flush=True)
    print(f"Base preprocessing: Resize{IMAGE_SIZE} + Normalize(mean=0.5,std=0.5)", flush=True)
    print(f"Training brightness/contrast jitter: ±{args.brightness_contrast:.2f}", flush=True)
    print("Checkpoint selection: mean per-patient Validation ROC-AUC", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            if use_amp:
                images = images.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
            else:
                images = images.to(device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images).view(-1)
                loss = loss_fn(logits.float(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item()) * len(labels)
            total_seen += len(labels)

        probabilities = predict_probabilities(model, valid_loader, device, use_amp)
        global_auc = safe_auc(y_valid, probabilities)
        macro_auc, patient_aucs = patient_macro_auc(valid_df, probabilities)
        fixed_metrics = metrics_at_threshold(y_valid, probabilities, 0.5)

        # Use patient-macro AUC for checkpoint selection, rather than tuning
        # a threshold on the same validation set every epoch.
        selection_score = macro_auc if np.isfinite(macro_auc) else global_auc
        scheduler.step(selection_score)

        print(
            f"Epoch {epoch:02d}/{args.epochs}"
            f" | Loss={total_loss/max(1,total_seen):.4f}"
            f" | Global AUC={global_auc:.4f}"
            f" | Patient-macro AUC={macro_auc:.4f}"
            f" | BA@0.5={fixed_metrics['balanced_accuracy']:.4f}"
            f" | LR={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        print(f"  Patient AUCs: {patient_aucs}", flush=True)

        if selection_score > best_score + 1e-5:
            best_score = selection_score
            best_epoch = epoch
            wait = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(
                    f"Early stopping after {args.patience} epochs without "
                    "improvement in patient-macro AUC.",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state, strict=True)
    final_probs = predict_probabilities(model, valid_loader, device, use_amp)
    final_global_auc = safe_auc(y_valid, final_probs)
    final_macro_auc, final_patient_aucs = patient_macro_auc(valid_df, final_probs)
    final_at_05 = metrics_at_threshold(y_valid, final_probs, 0.5)
    descriptive_threshold = descriptive_best_threshold(y_valid, final_probs)
    final_at_descriptive = metrics_at_threshold(
        y_valid, final_probs, descriptive_threshold
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "model_name": "vgg11",
        "image_size": list(IMAGE_SIZE),
        "resize_mode": "direct_resize",
        "threshold": 0.5,  # Fixed threshold; do not silently use validation-tuned threshold.
        "validation_descriptive_threshold": descriptive_threshold,
        "best_epoch": int(best_epoch),
        "best_validation_patient_macro_auc": float(best_score),
        "validation_global_auc": float(final_global_auc),
        "validation_patient_macro_auc": float(final_macro_auc),
        "validation_ba_at_05": float(final_at_05["balanced_accuracy"]),
        "mean": list(MEAN),
        "std": list(STD),
        "optuna_best_params": {"dropout": float(args.dropout)},
        "training_config": {
            "epochs_max": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "brightness_contrast": float(args.brightness_contrast),
            "sampler": "patient_class_balanced",
            "pretrained": False,
        },
        "split_sha256": split_hash,
        "threshold_selected_on_validation": False,
    }
    torch.save(checkpoint, output_path)

    prediction_path = output_path.with_name(
        output_path.stem + "_validation_predictions.csv"
    )
    predictions = valid_df[
        ["sample_id", "patient_group", "label"]
    ].copy()
    predictions["probability_present"] = final_probs
    predictions["prediction_at_0_5"] = (final_probs >= 0.5).astype(int)
    predictions["prediction_at_descriptive_threshold"] = (
        final_probs >= descriptive_threshold
    ).astype(int)
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    patient_rows = []
    for patient, group in predictions.groupby("patient_group", sort=True):
        y = group["label"].to_numpy(dtype=int)
        p = group["probability_present"].to_numpy(dtype=float)
        m = metrics_at_threshold(y, p, 0.5)
        patient_rows.append({
            "patient_group": patient,
            "n_images": len(group),
            "n_negative": int((y == 0).sum()),
            "n_positive": int((y == 1).sum()),
            "roc_auc": safe_auc(y, p),
            "balanced_accuracy_at_0_5": m["balanced_accuracy"],
            "sensitivity_at_0_5": m["sensitivity"],
            "specificity_at_0_5": m["specificity"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        })
    patient_metrics_path = output_path.with_name(
        output_path.stem + "_patient_metrics.csv"
    )
    pd.DataFrame(patient_rows).to_csv(
        patient_metrics_path, index=False, encoding="utf-8-sig"
    )

    summary = {
        "checkpoint": str(output_path),
        "best_epoch": int(best_epoch),
        "selection_metric": "mean per-patient Validation ROC-AUC",
        "validation_global_auc": float(final_global_auc),
        "validation_patient_macro_auc": float(final_macro_auc),
        "validation_patient_aucs": final_patient_aucs,
        "validation_at_threshold_0_5": final_at_05,
        "validation_descriptive_optimal_threshold": descriptive_threshold,
        "validation_at_descriptive_threshold": final_at_descriptive,
        "split_sha256": split_hash,
        "note": (
            "The descriptive threshold is optimized on this validation set and "
            "is optimistic. The saved inference threshold remains fixed at 0.5. "
            "Validation contains only a small number of held-out patients; "
            "private-test performance may differ."
        ),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Reload test: verifies the saved checkpoint can be reconstructed.
    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    check_model = build_model(args.dropout)
    check_model.load_state_dict(loaded["model_state_dict"], strict=True)
    check_model.eval()

    print("\n=== FINAL RESULT ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation global ROC-AUC: {final_global_auc:.4f}")
    print(f"Validation patient-macro ROC-AUC: {final_macro_auc:.4f}")
    print(f"Validation BA @ fixed 0.5: {final_at_05['balanced_accuracy']:.4f}")
    print(
        f"Descriptive validation threshold: {descriptive_threshold:.4f}"
        f" | BA={final_at_descriptive['balanced_accuracy']:.4f}"
    )
    print(f"Patient AUCs: {final_patient_aucs}")
    print("Confusion matrix @0.5 [true rows 0/1, predicted cols 0/1]:")
    print(np.array([
        [final_at_05["tn"], final_at_05["fp"]],
        [final_at_05["fn"], final_at_05["tp"]],
    ]))
    print(f"Saved checkpoint: {output_path}")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved patient metrics: {patient_metrics_path}")
    print(f"Saved summary: {summary_path}")
    print("Checkpoint reload test: OK")
    print("Original vgg.pt and vgg_optuna.pt were not modified.")


if __name__ == "__main__":
    main()
