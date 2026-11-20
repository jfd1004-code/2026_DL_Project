"""Task 2: VGG11 ultrasound sheath binary classification (no pretrained weights).

Training (run from project root):
    python code/vgg.py --mode train
Inference with a CSV containing a path column:
    python code/vgg.py --mode predict --csv data/T2_data/test_labels.csv

Requirements: torch, torchvision, pillow, pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent.parent
WEIGHT_PATH = ROOT / "vgg.pt"
SPLIT_PATH = ROOT / "task2_split.csv"
SEED = 42
IMAGE_SIZE = (160, 112)  # height, width; similar aspect ratio to 724x480
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_csv(arg: str | None, prediction: bool) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"CSV file not found: {p}")
        return p
    if prediction:
        raise ValueError("Prediction mode requires --csv (CSV with a path column).")
    candidates = [
        ROOT / "data" / "T2_data" / "labels.csv",
        ROOT / "T2_data" / "labels.csv",
        ROOT / "data" / "T2_data" / "T2_data" / "labels.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Cannot find T2_data/labels.csv. Extract T2_data.zip into "
        "DL_Term/data/T2_data/ or specify --csv PATH."
    )


def read_labels(csv_path: Path, require_targets: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sample_id": str, "patient_group": str})
    need = ["path"] + (["sample_id", "label", "patient_group"] if require_targets else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if df["path"].isna().any():
        raise ValueError("CSV contains empty image paths")
    if require_targets:
        if df[["sample_id", "label", "patient_group"]].isna().any().any():
            raise ValueError("CSV contains missing ids, labels, or patient groups")
        df["sample_id"] = df["sample_id"].astype(str)
        df["patient_group"] = df["patient_group"].astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError("Labels must be 0 or 1")
        if df["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample_id values found")
    return df.reset_index(drop=True)


def resolve_image_path(value: str, csv_path: Path) -> Path:
    relative = Path(str(value).replace("\\", "/"))
    if relative.is_absolute() and relative.is_file():
        return relative
    candidates = [
        csv_path.parent / relative,
        csv_path.parent / "images" / relative,
        csv_path.parent.parent / relative,
        ROOT / relative,
        ROOT / "data" / relative,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Image not found: {value!r}; checked relative to {csv_path.parent}"
    )


def get_transforms(training: bool, image_size=IMAGE_SIZE):
    operations = [transforms.Resize(image_size)]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


class UltrasoundDataset(Dataset):
    def __init__(self, df: pd.DataFrame, csv_path: Path, training: bool,
                 image_size=IMAGE_SIZE):
        self.files = [resolve_image_path(s, csv_path) for s in df["path"].tolist()]
        self.labels = df["label"].to_numpy(dtype=np.float32) if "label" in df else None
        self.transform = get_transforms(training, image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as im:
            image = self.transform(im.convert("RGB"))
        if self.labels is None:
            return image
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)


def build_model() -> nn.Module:
    # Official torchvision VGG11 convolutional architecture, without ImageNet weights.
    # init_weights=False avoids costly initialization of its original giant classifier,
    # which is replaced immediately with a smaller binary classification head.
    model = models.vgg11(weights=None, init_weights=False)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)
    return model


def load_or_create_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if SPLIT_PATH.exists():
        split_df = pd.read_csv(SPLIT_PATH, dtype={"sample_id": str})
        if (not {"sample_id", "split"}.issubset(split_df.columns)
                or split_df["sample_id"].duplicated().any()
                or set(split_df["sample_id"]) != set(df["sample_id"])):
            raise ValueError(
                "Existing task2_split.csv is incompatible with the current labels. "
                "Check its contents before manually deleting or replacing it."
            )
        split_map = dict(zip(split_df["sample_id"], split_df["split"]))
        split_names = df["sample_id"].map(split_map)
        if not set(split_names.unique()).issubset({"train", "valid"}):
            raise ValueError("Split CSV must contain only 'train' and 'valid'")
        tr = np.flatnonzero(split_names.to_numpy() == "train")
        va = np.flatnonzero(split_names.to_numpy() == "valid")
    else:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = []
        overall_pos_rate = df["label"].mean()
        for tr, va in sgkf.split(df, df["label"], groups=df["patient_group"]):
            if df.iloc[va]["label"].nunique() != 2 or df.iloc[tr]["label"].nunique() != 2:
                continue
            score = (abs(df.iloc[va]["label"].mean() - overall_pos_rate)
                     + 0.1 * abs(len(va) / len(df) - 0.2))
            folds.append((score, tr, va))
        if not folds:
            raise ValueError("Could not find a patient-group split containing both labels")
        _, tr, va = min(folds, key=lambda item: item[0])
        assignments = np.full(len(df), "train", dtype=object)
        assignments[va] = "valid"
        pd.DataFrame({"sample_id": df["sample_id"], "split": assignments}).to_csv(
            SPLIT_PATH, index=False
        )
        print(f"Saved patient-wise split: {SPLIT_PATH}")

    train_groups = set(df.iloc[tr]["patient_group"])
    valid_groups = set(df.iloc[va]["patient_group"])
    if train_groups & valid_groups:
        raise ValueError("Patient group leaked between train and valid")
    if not len(tr) or not len(va) or df.iloc[tr]["label"].nunique() != 2 or df.iloc[va]["label"].nunique() != 2:
        raise ValueError("Train and valid must both contain positive and negative examples")
    print(f"Train: {len(tr)} images / {len(train_groups)} patients"
          f" | Valid: {len(va)} images / {len(valid_groups)} patients")
    print(f"Positive: train={int(df.iloc[tr]['label'].sum())}, valid={int(df.iloc[va]['label'].sum())}")
    return tr, va


@torch.no_grad()
def probability_predictions(model, loader, device) -> np.ndarray:
    model.eval()
    pieces = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = model(images.to(device)).view(-1)
        pieces.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)


def best_threshold_and_ba(y_true: np.ndarray, probs: np.ndarray):
    # Validation-only tuning. The same chosen threshold is stored for inference.
    candidates = np.linspace(0.10, 0.90, 33)
    candidates = sorted(candidates, key=lambda t: abs(t - 0.5))
    best_ba, best_threshold = -1.0, 0.5
    for t in candidates:
        ba = balanced_accuracy_score(y_true, (probs >= t).astype(int))
        if ba > best_ba + 1e-9:
            best_ba, best_threshold = float(ba), float(t)
    return best_threshold, best_ba


def train(csv_path: Path, epochs: int, batch_size: int, lr: float,
          patience: int, image_size=IMAGE_SIZE):
    set_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_labels(csv_path, require_targets=True)
    tr, va = load_or_create_split(df)
    train_df, valid_df = df.iloc[tr], df.iloc[va]
    train_ds = UltrasoundDataset(train_df, csv_path, training=True, image_size=image_size)
    valid_ds = UltrasoundDataset(valid_df, csv_path, training=False, image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    positive_weight = n_neg / n_pos
    model = build_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    print(f"Device: {device} | VGG11 from scratch | Image size (H,W): {image_size}", flush=True)
    print(f"Positive class weight: {positive_weight:.3f}")
    best_ba, best_epoch, wait = -1.0, -1, 0
    y_valid = valid_df["label"].to_numpy(dtype=int)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)
        probs = probability_predictions(model, valid_loader, device)
        ba_at_half = balanced_accuracy_score(y_valid, (probs >= 0.5).astype(int))
        threshold, tuned_ba = best_threshold_and_ba(y_valid, probs)
        scheduler.step(tuned_ba)
        print(f"Epoch {epoch:02d}/{epochs} | Loss {running_loss/len(train_ds):.4f}"
              f" | BA@0.5 {ba_at_half:.4f} | Best Val BA {tuned_ba:.4f}"
              f" | Threshold {threshold:.3f} | LR {optimizer.param_groups[0]['lr']:.6f}", flush=True)
        if tuned_ba > best_ba + 1e-6:
            best_ba, best_epoch, wait = tuned_ba, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "vgg11",
                "image_size": list(image_size),
                "threshold": float(threshold),
                "best_valid_ba": float(best_ba),
                "best_epoch": int(best_epoch),
                "mean": list(MEAN),
                "std": list(STD),
            }, WEIGHT_PATH)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    verified_model = build_model()
    verified_model.load_state_dict(checkpoint["model_state_dict"])
    verified_model.eval()
    print("\n=== Best Model ===")
    print(f"Best Epoch: {checkpoint['best_epoch']}")
    print(f"Best Validation Balanced Accuracy: {checkpoint['best_valid_ba']:.4f}")
    print(f"Best Threshold: {checkpoint['threshold']:.3f}")
    print(f"Saved: {WEIGHT_PATH}")
    print("Checkpoint load test: OK")
    print("Note: threshold selected on validation data; validation BA is optimistic.")


def predict(csv_path: Path, output: Path, batch_size: int):
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Missing trained weights: {WEIGHT_PATH}")
    ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    df = read_labels(csv_path, require_targets=False)
    ds = UltrasoundDataset(df, csv_path, training=False,
                           image_size=tuple(ckpt["image_size"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = probability_predictions(model, loader, torch.device("cpu"))
    threshold = float(ckpt["threshold"])
    result = pd.DataFrame({
        "probability_present": probs,
        "prediction": (probs >= threshold).astype(int),
    })
    if "sample_id" in df:
        result.insert(0, "sample_id", df["sample_id"].astype(str))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved predictions: {output} ({len(result)} images, threshold={threshold:.3f})")
    if "label" in df and df["label"].nunique() == 2:
        y = df["label"].astype(int).to_numpy()
        print(f"Balanced Accuracy on input CSV: "
              f"{balanced_accuracy_score(y, result['prediction']):.4f}")
        print("Confusion matrix [true rows 0/1, predicted cols 0/1]:")
        print(confusion_matrix(y, result["prediction"], labels=[0, 1]))


def main():
    parser = argparse.ArgumentParser(description="Train or use Task 2 VGG11")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--csv", default=None, help="path to labels.csv or prediction input CSV")
    parser.add_argument("--output", default=str(ROOT / "vgg_predictions.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    csv_path = find_csv(args.csv, prediction=args.mode == "predict")
    if args.mode == "train":
        train(csv_path, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, patience=args.patience)
    else:
        predict(csv_path, Path(args.output), batch_size=args.batch_size)


if __name__ == "__main__":
    print("[VGG11] script started — loading dataset...", flush=True)
    main()
