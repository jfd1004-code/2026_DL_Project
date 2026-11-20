"""Task 1: HELOC risk prediction with a fully connected DNN.

Training (from project root):
    python code/task1_robust_experiment.py

Inference:
    python code/task1_robust_experiment.py --mode predict --csv path/to/new_data.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Code location: DL_Term/code/task1_robust_experiment.py
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "T1_data"
WEIGHTS_PATH = PROJECT_DIR / "task1_robust.pt"
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 7
MIN_DELTA = 0.0001
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
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
            f"데이터 폴더가 없습니다: {DATA_DIR}\n"
            "CSV 파일을 DL_Term/data/T1_data 폴더에 넣어 주세요."
        )
    matches = []
    for path in sorted(DATA_DIR.rglob("*.csv")):
        try:
            headers = pd.read_csv(path, nrows=0).columns
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        if "is_at_risk" in headers and "sample_id" in headers:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"'is_at_risk'와 'sample_id' 열이 있는 CSV를 찾지 못했습니다: {DATA_DIR}"
        )
    if len(matches) > 1:
        raise ValueError(
            "학습용 CSV가 여러 개 발견되었습니다. --csv 옵션으로 하나를 지정하세요:\n"
            + "\n".join(str(x) for x in matches)
        )
    return matches[0]


def resolve_csv_path(value: str | None) -> Path:
    if not value:
        return find_training_csv()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = PROJECT_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {candidate}")
    return candidate


def fit_preprocessing(x_train_raw: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Fit special-code indicators, median imputation, RobustScaler on TRAIN only."""
    raw = x_train_raw[feature_columns].apply(pd.to_numeric, errors="coerce")

    # Keep the same indicator policy as the best baseline DNN.
    indicator_specs = []
    for col in feature_columns:
        for code in SPECIAL_CODES:
            if raw[col].eq(code).any():
                indicator_specs.append([col, int(code)])

    numerical = raw.replace(list(SPECIAL_CODES), np.nan)
    numerical = numerical.replace([np.inf, -np.inf], np.nan)
    medians = numerical.median().fillna(0.0)
    filled = numerical.fillna(medians)

    # The only experimental change: RobustScaler instead of StandardScaler.
    # RobustScaler centers on the median and scales by the IQR (Q3 - Q1).
    # This does NOT clip or delete large values.
    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(filled)

    return {
        "feature_columns": list(feature_columns),
        "indicator_specs": indicator_specs,
        "medians": [float(medians[c]) for c in feature_columns],
        "centers": [float(v) for v in scaler.center_],
        "scales": [float(v) for v in scaler.scale_],
        "scaler_type": "RobustScaler",
    }


def transform_features(raw_df: pd.DataFrame, prep: dict) -> np.ndarray:
    """Use the same train-derived preprocessing on validation/test data."""
    feature_columns = prep["feature_columns"]
    missing = [c for c in feature_columns if c not in raw_df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 필요한 열이 없습니다: {missing}")

    raw = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    indicators = [
        raw[col].eq(int(code)).to_numpy(dtype=np.float32)
        for col, code in prep["indicator_specs"]
    ]

    numerical = raw.replace(list(SPECIAL_CODES), np.nan)
    numerical = numerical.replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(prep["medians"], index=feature_columns)
    numerical = numerical.fillna(medians)

    values = numerical.to_numpy(dtype=np.float32)
    centers = np.asarray(prep["centers"], dtype=np.float32)
    scales = np.asarray(prep["scales"], dtype=np.float32)
    scaled = (values - centers) / scales

    if indicators:
        scaled = np.column_stack([scaled, *indicators])
    return np.asarray(scaled, dtype=np.float32)


def create_model(input_dim: int) -> nn.Module:
    """Fully connected DNN: input -> 128 (BN) -> 64 (BN) -> 1."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def train(csv_path: Path) -> None:
    set_seed()
    device = get_device()
    df = pd.read_csv(csv_path)
    if "is_at_risk" not in df.columns:
        raise ValueError("학습 CSV에 'is_at_risk' 정답 열이 없습니다.")
    if "sample_id" not in df.columns:
        raise ValueError("학습 CSV에 'sample_id' 열이 없습니다.")

    feature_columns = [c for c in df.columns if c not in ("sample_id", "is_at_risk")]
    if len(feature_columns) != 23:
        raise ValueError(
            f"과제 데이터는 입력 Feature 23개여야 하지만 {len(feature_columns)}개를 발견했습니다. "
            "CSV 파일과 열 이름을 확인하세요."
        )

    y = pd.to_numeric(df["is_at_risk"], errors="raise").to_numpy(dtype=np.float32)
    if np.isnan(y).any() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("is_at_risk 열은 0과 1만 포함해야 합니다.")

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        df[feature_columns], y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    prep = fit_preprocessing(x_train_raw, feature_columns)
    x_train = transform_features(x_train_raw, prep)
    x_val = transform_features(x_val_raw, prep)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train).reshape(-1, 1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).reshape(-1, 1).to(device)

    model = create_model(x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("=== Task 1 DNN Training ===")
    print(f"CSV: {csv_path}")
    print(f"Device: {device}")
    print(f"Train: {len(y_train)} | Valid: {len(y_val)}")
    print(f"Inputs: {len(feature_columns)} original + {len(prep['indicator_specs'])} flags = {x_train.shape[1]}")
    print("Scaler: RobustScaler (Train median / interquartile range; no clipping)")

    best_auc = -float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        train_loss = running_loss / seen

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_prob = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob))

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f}"
        )

        if val_auc > best_auc + MIN_DELTA:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            # Put weights AND preprocessing in one task1.pt file.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(x_train.shape[1]),
                    "preprocessing": prep,
                    "best_epoch": int(best_epoch),
                    "best_auc": float(best_auc),
                },
                WEIGHTS_PATH,
            )
            print(f"  -> Best model saved: {WEIGHTS_PATH.name}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early Stopping at Epoch {epoch}")
                break

    print("\n=== Best Model ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation AUROC: {best_auc:.4f}")
    print(f"Saved: {WEIGHTS_PATH}")
    # Verify that the exact saved checkpoint can be loaded.
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    reloaded = create_model(int(checkpoint["input_dim"]))
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    print("Checkpoint load test: OK")


def predict_proba(input_df: pd.DataFrame, checkpoint_path: Path = WEIGHTS_PATH) -> np.ndarray:
    """Return P(is_at_risk=1) for any dataframe with the 23 input columns."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_model(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x = transform_features(input_df, checkpoint["preprocessing"])
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        return torch.sigmoid(logits).numpy().reshape(-1)


def predict(csv_path: Path, output_path: Path) -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"먼저 학습해서 {WEIGHTS_PATH.name} 파일을 만들어야 합니다.")
    df = pd.read_csv(csv_path)
    probabilities = predict_proba(df)
    result = pd.DataFrame({"is_at_risk_probability": probabilities})
    if "sample_id" in df.columns:
        result.insert(0, "sample_id", df["sample_id"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predicted {len(result)} rows; saved: {output_path}")
    if "is_at_risk" in df.columns:
        y = pd.to_numeric(df["is_at_risk"], errors="coerce")
        if y.notna().all() and y.nunique() == 2:
            print("주의: 입력 CSV에 정답이 있습니다. 학습 데이터에서 측정한 AUC는 Test 성능이 아닙니다.")
            print(f"Input-data AUROC: {roc_auc_score(y, probabilities):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 RobustScaler DNN experiment and predictor")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--csv", type=str, default=None, help="input CSV path")
    parser.add_argument("--output", type=str, default=None, help="prediction output CSV path")
    args = parser.parse_args()
    if args.mode == "train":
        train(resolve_csv_path(args.csv))
    else:
        if not args.csv:
            parser.error("predict 모드에서는 --csv 파일 경로가 필요합니다.")
        output_path = Path(args.output) if args.output else PROJECT_DIR / "task1_predictions.csv"
        if not output_path.is_absolute():
            output_path = PROJECT_DIR / output_path
        predict(resolve_csv_path(args.csv), output_path)


if __name__ == "__main__":
    main()
