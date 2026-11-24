"""Task 1: HELOC credit-risk prediction with a fully connected DNN.

Train:
    python code/task1.py --mode train --csv data/T1_data/heloc_release.csv

Predict:
    python code/task1.py --mode predict --csv path/to/unlabeled.csv --output predictions.csv

The final checkpoint stores both the DNN weights and all train-fitted
preprocessing parameters. It outputs P(is_at_risk=1), evaluated by AUROC.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "T1_data"
WEIGHTS_PATH = PROJECT_DIR / "task1.pt"

SEED = 42
VALID_SIZE = 0.20
MAX_EPOCHS = 60
PATIENCE = 9
MIN_DELTA = 1e-4
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10
# Median of best epochs from the three-seed validation comparison.
FINAL_FIT_EPOCHS = 17
SPECIAL_CODES = (-7, -8, -9)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_training_csv() -> Path:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            "Extract T1_data.zip into data/T1_data/ or pass --csv."
        )
    matches = []
    for path in sorted(DATA_DIR.rglob("*.csv")):
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        if {"sample_id", "is_at_risk"}.issubset(columns):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No training CSV containing sample_id and is_at_risk in {DATA_DIR}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple training CSVs found; specify one with --csv:\n"
            + "\n".join(str(p) for p in matches)
        )
    return matches[0]


def resolve_csv_path(value: str | None) -> Path:
    if value is None:
        return find_training_csv()
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return path


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [c for c in df.columns if c not in ("sample_id", "is_at_risk")]
    if len(columns) != 23:
        raise ValueError(
            f"Expected 23 input features, found {len(columns)}. "
            "Check that sample_id and is_at_risk are the only excluded columns."
        )
    return columns


def fit_preprocessing(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Fit sentinel flags, median imputation and RobustScaler on Train only."""
    raw = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    # Keep separate indicators because -7, -8 and -9 have different meanings.
    indicator_specs: list[list[Any]] = []
    for column in feature_columns:
        for code in SPECIAL_CODES:
            if raw[column].eq(code).any():
                indicator_specs.append([column, int(code)])

    numeric = raw.replace(list(SPECIAL_CODES), np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median().fillna(0.0)
    filled = numeric.fillna(medians)

    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(filled)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0

    return {
        "feature_columns": list(feature_columns),
        "indicator_specs": indicator_specs,
        "medians": [float(medians[c]) for c in feature_columns],
        "centers": [float(v) for v in scaler.center_],
        "scales": [float(v) for v in scales],
        "scaler_type": "RobustScaler",
        "special_codes": list(SPECIAL_CODES),
    }


def transform_features(raw_df: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    """Apply the saved Train-fitted preprocessing to any split."""
    feature_columns = prep["feature_columns"]
    missing = [c for c in feature_columns if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    raw = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    indicators = [
        raw[column].eq(int(code)).to_numpy(dtype=np.float32)
        for column, code in prep["indicator_specs"]
    ]

    numeric = raw.replace(prep["special_codes"], np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(prep["medians"], index=feature_columns, dtype=np.float64)
    numeric = numeric.fillna(medians)

    values = numeric.to_numpy(dtype=np.float32)
    centers = np.asarray(prep["centers"], dtype=np.float32)
    scales = np.asarray(prep["scales"], dtype=np.float32)
    scaled = (values - centers) / scales

    if indicators:
        scaled = np.column_stack([scaled, *indicators])
    return np.asarray(scaled, dtype=np.float32)


class CreditRiskDNN(nn.Module):
    """Compact fully connected network selected by repeated validation runs."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train(csv_path: Path, seed: int = SEED) -> None:
    set_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = get_device()

    df = pd.read_csv(csv_path)
    if "is_at_risk" not in df.columns or "sample_id" not in df.columns:
        raise ValueError("Training CSV must contain sample_id and is_at_risk.")

    feature_columns = get_feature_columns(df)
    y = pd.to_numeric(df["is_at_risk"], errors="raise").to_numpy(dtype=np.float32)
    if np.isnan(y).any() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("is_at_risk must contain only 0 and 1.")

    # Stage 1: hold out validation rows to select a sensible training duration.
    train_raw, valid_raw, y_train, y_valid = train_test_split(
        df[feature_columns], y, test_size=VALID_SIZE,
        random_state=seed, stratify=y,
    )
    prep_valid = fit_preprocessing(train_raw, feature_columns)
    x_train = transform_features(train_raw, prep_valid)
    x_valid = transform_features(valid_raw, prep_valid)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )

    model = CreditRiskDNN(x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5,
    )

    x_valid_tensor = torch.from_numpy(x_valid).to(device)
    y_valid_tensor = torch.from_numpy(y_valid.reshape(-1, 1)).to(device)
    best_auc = -1.0
    best_epoch = 0
    no_improve = 0

    print("=== Task 1: optimized DNN ===", flush=True)
    print(f"CSV: {csv_path}")
    print(f"Device: {device}")
    print(f"Train / Valid: {len(y_train)} / {len(y_valid)}")
    print(f"Features: {len(feature_columns)} original + "
          f"{len(prep_valid['indicator_specs'])} sentinel indicators = {x_train.shape[1]}")
    print(f"Positive rate: train={y_train.mean():.4f}, valid={y_valid.mean():.4f}")
    print("Preprocessing: sentinel indicators + median imputation + RobustScaler")
    print("Architecture: input -> 64 -> 32 -> 1 | BN + ReLU + Dropout(0.10)")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)

        model.eval()
        with torch.no_grad():
            valid_logits = model(x_valid_tensor)
            valid_loss = float(criterion(valid_logits, y_valid_tensor).item())
            valid_prob = torch.sigmoid(valid_logits).cpu().numpy().reshape(-1)

        valid_auc = float(roc_auc_score(y_valid, valid_prob))
        scheduler.step(valid_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | Train Loss {running_loss / seen:.4f} | "
            f"Valid Loss {valid_loss:.4f} | Valid AUROC {valid_auc:.5f} | "
            f"LR {current_lr:.6g}", flush=True,
        )

        if valid_auc > best_auc + MIN_DELTA:
            best_auc = valid_auc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    if best_epoch == 0:
        raise RuntimeError("Validation training did not produce a best epoch.")

    # Stage 2: refit one DNN on every labeled training row.
    # The final epoch count is the median best epoch from the three-seed
    # comparison documented in TASK1_OPTIMIZATION_REPORT.md.
    final_epochs = FINAL_FIT_EPOCHS
    prep_full = fit_preprocessing(df[feature_columns], feature_columns)
    x_full = transform_features(df[feature_columns], prep_full)
    full_dataset = TensorDataset(
        torch.from_numpy(x_full), torch.from_numpy(y.reshape(-1, 1)),
    )
    full_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1000), drop_last=False,
    )

    set_seed(seed + 1000)
    final_model = CreditRiskDNN(x_full.shape[1]).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer, T_max=final_epochs, eta_min=1e-5,
    )

    print(
        f"\nRefitting final DNN on all {len(y)} labeled rows for "
        f"{final_epochs} epochs...", flush=True,
    )
    for epoch in range(1, final_epochs + 1):
        final_model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in full_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            final_optimizer.zero_grad(set_to_none=True)
            logits = final_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=2.0)
            final_optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        final_scheduler.step()
        print(
            f"Full-data epoch {epoch:02d}/{final_epochs} | "
            f"Loss {running_loss / seen:.4f} | "
            f"LR {final_optimizer.param_groups[0]['lr']:.6g}", flush=True,
        )

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "input_dim": int(x_full.shape[1]),
            "preprocessing": prep_full,
            "model_config": {
                "hidden_dims": [64, 32],
                "dropout": DROPOUT,
                "batch_norm": True,
            },
            "seed": int(seed + 1000),
            "best_epoch": int(final_epochs),
            "validation_reference_auc": float(best_auc),
            "validation_reference_epoch": int(best_epoch),
            "validation_seed": int(seed),
            "trained_on_all_rows": True,
            "train_rows": int(len(y)),
        },
        WEIGHTS_PATH,
    )

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    reloaded = CreditRiskDNN(int(checkpoint["input_dim"]))
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()

    print("\n=== Final submission checkpoint ===")
    print(f"Validation reference AUROC: {best_auc:.5f} (epoch {best_epoch})")
    print(f"Final fit: {checkpoint['train_rows']} rows / {checkpoint['best_epoch']} epochs")
    print(f"Final weights: {WEIGHTS_PATH}")
    print("Checkpoint reload test: OK")


def predict_proba(input_df: pd.DataFrame, checkpoint_path: Path = WEIGHTS_PATH) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CreditRiskDNN(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x = transform_features(input_df, checkpoint["preprocessing"])
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        return torch.sigmoid(logits).numpy().reshape(-1)


def predict(csv_path: Path, output_path: Path) -> None:
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Trained weights not found: {WEIGHTS_PATH}")
    df = pd.read_csv(csv_path)
    probabilities = predict_proba(df)
    result = pd.DataFrame({"is_at_risk_probability": probabilities})
    if "sample_id" in df.columns:
        result.insert(0, "sample_id", df["sample_id"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")
    if "is_at_risk" in df.columns:
        labels = pd.to_numeric(df["is_at_risk"], errors="coerce")
        if labels.notna().all() and labels.nunique() == 2:
            print("AUROC on supplied labeled CSV: "
                  f"{roc_auc_score(labels.astype(int), probabilities):.5f}")
            print("Warning: this is not a private-test score unless the CSV is the official test set.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 DNN training and inference")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--csv", default=None, help="input CSV path")
    parser.add_argument("--output", default=None, help="prediction output CSV path")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv)
    if args.mode == "train":
        train(csv_path, seed=args.seed)
    else:
        output = Path(args.output) if args.output else PROJECT_DIR / "task1_predictions.csv"
        if not output.is_absolute():
            output = PROJECT_DIR / output
        predict(csv_path, output)


if __name__ == "__main__":
    main()
