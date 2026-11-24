"""
Task 1 HELOC: Optuna search for ONE final fully-connected DNN.

- Uses only the provided HELOC CSV.
- Keeps -7/-8/-9 as separate semantic indicators and fits imputation/scaling
  separately inside each training fold.
- Uses stratified 3-fold CV for model selection.
- Searches only fully-connected DNN configurations.
- No tree model, no ensemble, no external data, no other neural feature model.
- Saves ONE final DNN checkpoint to task1_optuna.pt.
- The checkpoint is compatible with this script's --mode predict.

Run from project root:
python code/task1_optuna_cv.py --mode search --csv data/T1_data/heloc_release.csv

Prediction:
python code/task1_optuna_cv.py --mode predict --csv path/to/unlabeled.csv \
  --checkpoint task1_optuna.pt --output task1_optuna_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SPECIAL_CODES = (-7, -8, -9)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_DIR / "data" / "T1_data" / "heloc_release.csv"
DEFAULT_CHECKPOINT = PROJECT_DIR / "task1_optuna.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {"sample_id", "is_at_risk"}.issubset(df.columns):
        raise ValueError("CSV must contain sample_id and is_at_risk.")
    feature_columns = [c for c in df.columns if c not in ("sample_id", "is_at_risk")]
    if len(feature_columns) != 23:
        raise ValueError(f"Expected 23 features, found {len(feature_columns)}.")
    y = pd.to_numeric(df["is_at_risk"], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("is_at_risk must contain only 0 and 1.")
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id contains duplicates.")
    return df, feature_columns, y


def fit_preprocessing(train_df: pd.DataFrame, feature_columns: list[str],
                      scaler_name: str) -> dict[str, Any]:
    raw = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    indicator_specs = []
    for col in feature_columns:
        for code in SPECIAL_CODES:
            if raw[col].eq(code).any():
                indicator_specs.append([col, int(code)])

    numeric = raw.replace(list(SPECIAL_CODES), np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median().fillna(0.0)
    filled = numeric.fillna(medians)

    if scaler_name == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(filled)
        centers = np.asarray(scaler.center_, dtype=np.float64)
        scales = np.asarray(scaler.scale_, dtype=np.float64)
    elif scaler_name == "standard":
        scaler = StandardScaler().fit(filled)
        centers = np.asarray(scaler.mean_, dtype=np.float64)
        scales = np.asarray(scaler.scale_, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")

    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return {
        "feature_columns": list(feature_columns),
        "indicator_specs": indicator_specs,
        "medians": [float(medians[c]) for c in feature_columns],
        "centers": [float(v) for v in centers],
        "scales": [float(v) for v in scales],
        "scaler_type": scaler_name,
        "special_codes": list(SPECIAL_CODES),
    }


def transform_features(frame: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    cols = prep["feature_columns"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw = frame[cols].apply(pd.to_numeric, errors="coerce")
    indicators = [
        raw[col].eq(int(code)).to_numpy(dtype=np.float32)
        for col, code in prep["indicator_specs"]
    ]
    numeric = raw.replace(prep["special_codes"], np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(prep["medians"], index=cols, dtype=np.float64)
    values = numeric.fillna(medians).to_numpy(dtype=np.float32)

    scaled = (
        values - np.asarray(prep["centers"], dtype=np.float32)
    ) / np.asarray(prep["scales"], dtype=np.float32)

    if indicators:
        scaled = np.column_stack([scaled, *indicators])
    return np.asarray(scaled, dtype=np.float32)


class CreditRiskDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int],
                 dropout: float, norm: str, activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(current, width))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(width))
            elif norm == "layer":
                layers.append(nn.LayerNorm(width))
            elif norm != "none":
                raise ValueError(f"Unknown norm: {norm}")

            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = width

        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def make_model(input_dim: int, params: dict[str, Any]):
    return CreditRiskDNN(
        input_dim=input_dim,
        hidden_dims=list(params["hidden_dims"]),
        dropout=float(params["dropout"]),
        norm=str(params["norm"]),
        activation=str(params["activation"]),
    )


def combined_loss(logits, targets, ranking_weight: float):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    if ranking_weight <= 0:
        return bce

    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1)
    positive = flat_logits[flat_targets > 0.5]
    negative = flat_logits[flat_targets < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return bce

    # Pairwise logistic ranking objective: encourage positive scores
    # to be larger than negative scores, matching the AUROC goal.
    pairwise = F.softplus(
        -(positive[:, None] - negative[None, :])
    ).mean()
    return (1.0 - ranking_weight) * bce + ranking_weight * pairwise


def evaluate_auc(model, x_valid, y_valid, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_valid).to(device)).reshape(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return float(roc_auc_score(y_valid, probs))


def train_fold(train_df, valid_df, feature_columns, params, seed,
               max_epochs, patience, device):
    set_seed(seed)
    prep = fit_preprocessing(train_df, feature_columns, params["scaler"])
    x_train = transform_features(train_df, prep)
    x_valid = transform_features(valid_df, prep)
    y_train = train_df["_target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_valid = valid_df["_target"].to_numpy(dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    model = make_model(x_train.shape[1], params).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    best_auc = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = combined_loss(
                logits, batch_y, float(params["ranking_weight"])
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        auc = evaluate_auc(model, x_valid, y_valid, device)
        scheduler.step(auc)

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return best_auc, best_epoch


def suggest_params(trial):
    depth = trial.suggest_categorical("depth", [2, 3])
    width1 = trial.suggest_categorical("width1", [64, 128, 256])
    width2 = trial.suggest_categorical("width2", [32, 64, 128])
    hidden_dims = [width1, width2]
    if depth == 3:
        hidden_dims.append(trial.suggest_categorical("width3", [16, 32, 64]))

    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "norm": trial.suggest_categorical("norm", ["batch", "layer", "none"]),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "scaler": trial.suggest_categorical("scaler", ["robust", "standard"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "ranking_weight": trial.suggest_float("ranking_weight", 0.0, 0.5),
    }


def params_for_baseline():
    return {
        "depth": 2,
        "width1": 64,
        "width2": 32,
        "dropout": 0.10,
        "norm": "batch",
        "activation": "relu",
        "scaler": "robust",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "ranking_weight": 0.0,
    }


def convert_trial_params(raw):
    depth = int(raw["depth"])
    dims = [int(raw["width1"]), int(raw["width2"])]
    if depth == 3:
        dims.append(int(raw["width3"]))
    return {
        "hidden_dims": dims,
        "dropout": float(raw["dropout"]),
        "norm": raw["norm"],
        "activation": raw["activation"],
        "scaler": raw["scaler"],
        "lr": float(raw["lr"]),
        "weight_decay": float(raw["weight_decay"]),
        "batch_size": int(raw["batch_size"]),
        "ranking_weight": float(raw["ranking_weight"]),
    }


def final_fit(df, feature_columns, params, epochs, device, output_path):
    prep = fit_preprocessing(df, feature_columns, params["scaler"])
    x = transform_features(df, prep)
    y = pd.to_numeric(df["is_at_risk"], errors="raise").to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(2026),
        drop_last=False,
    )

    set_seed(2026)
    model = make_model(x.shape[1], params).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = combined_loss(
                logits, batch_y, float(params["ranking_weight"])
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        scheduler.step()
        print(
            f"Final fit epoch {epoch:02d}/{epochs} | "
            f"loss={total_loss / max(1, seen):.5f}",
            flush=True,
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "preprocessing": prep,
        "model_config": {
            "hidden_dims": list(params["hidden_dims"]),
            "dropout": float(params["dropout"]),
            "norm": params["norm"],
            "activation": params["activation"],
        },
        "training_config": params,
        "final_epochs": int(epochs),
        "trained_on_all_rows": True,
        "train_rows": int(len(df)),
        "selection_metric": "mean 3-fold stratified validation AUROC",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    # Reload smoke test.
    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    check_params = {
        "hidden_dims": loaded["model_config"]["hidden_dims"],
        "dropout": loaded["model_config"]["dropout"],
        "norm": loaded["model_config"]["norm"],
        "activation": loaded["model_config"]["activation"],
    }
    check_model = make_model(loaded["input_dim"], check_params)
    check_model.load_state_dict(loaded["model_state_dict"], strict=True)
    check_model.eval()
    print(f"Saved final single-DNN checkpoint: {output_path}")
    print("Checkpoint reload test: OK")


def run_search(args):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Run: python -m pip install optuna"
        ) from exc

    csv_path = Path(args.csv).expanduser().resolve()
    df, feature_columns, y = load_data(csv_path)
    df = df.copy()
    df["_target"] = y

    device = choose_device(args.device)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    folds = list(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=2026
    ).split(df, y))

    print("=== TASK 1 OPTUNA 3-FOLD SEARCH ===")
    print(f"Device: {device}")
    print(f"Rows: {len(df)} | Features: {len(feature_columns)}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    print("CV: StratifiedKFold(3), preprocessing fitted separately in each fold")
    print("Final model: one fully-connected DNN; no ensemble")

    def objective(trial):
        params = suggest_params(trial)
        fold_aucs = []
        fold_epochs = []
        for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            auc, best_epoch = train_fold(
                train_df, valid_df, feature_columns, params,
                seed=2026 + fold_no, max_epochs=args.max_epochs,
                patience=args.patience, device=device,
            )
            fold_aucs.append(auc)
            fold_epochs.append(best_epoch)
            running_mean = float(np.mean(fold_aucs))
            trial.report(running_mean, step=fold_no)
            print(
                f"Trial {trial.number:02d} fold {fold_no}/3 "
                f"AUC={auc:.5f}, best_epoch={best_epoch}, "
                f"running_mean={running_mean:.5f}",
                flush=True,
            )
        mean_auc = float(np.mean(fold_aucs))
        trial.set_user_attr("fold_aucs", fold_aucs)
        trial.set_user_attr("best_epochs", fold_epochs)
        trial.set_user_attr("std_auc", float(np.std(fold_aucs)))
        return mean_auc

    study = optuna.create_study(
        direction="maximize",
        study_name="task1_heloc_dnn_cv",
        sampler=optuna.samplers.TPESampler(seed=2026),
    )

    # Ensure the current proven configuration is included as the baseline.
    study.enqueue_trial(params_for_baseline())
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    best_params = convert_trial_params(best.params)
    best_epochs = best.user_attrs.get("best_epochs", [17, 17, 17])
    final_epochs = max(1, int(np.median(best_epochs)))

    result = {
        "best_cv_mean_auc": float(best.value),
        "best_cv_std_auc": float(best.user_attrs.get("std_auc", float("nan"))),
        "fold_aucs": best.user_attrs.get("fold_aucs", []),
        "best_epochs_per_fold": best_epochs,
        "final_epochs": final_epochs,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "cv": "StratifiedKFold(3), random_state=2026",
        "note": (
            "CV score is used for model selection and is not the private-test "
            "AUROC. Selecting the best among trials can make the reported CV "
            "score optimistic. Final checkpoint is one DNN, not an ensemble."
        ),
    }
    result_path = Path(args.output_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== BEST TASK 1 CONFIGURATION ===")
    print(f"Mean 3-fold AUROC: {best.value:.5f}")
    print(f"Fold AUROCs: {result['fold_aucs']}")
    print(f"Fold best epochs: {best_epochs}")
    print(f"Final fit epochs: {final_epochs}")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"Saved search summary: {result_path}")

    if args.fit_final:
        output_path = Path(args.checkpoint).expanduser().resolve()
        final_fit(df, feature_columns, best_params, final_epochs, device, output_path)


def run_predict(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    frame = pd.read_csv(csv_path)
    prep = checkpoint["preprocessing"]
    x = transform_features(frame, prep)

    config = checkpoint["model_config"]
    params = {
        "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
        "norm": config["norm"],
        "activation": config["activation"],
    }
    model = make_model(checkpoint["input_dim"], params)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probabilities = torch.sigmoid(logits).numpy().reshape(-1)

    result = pd.DataFrame({
        "is_at_risk_probability": probabilities
    })
    if "sample_id" in frame.columns:
        result.insert(0, "sample_id", frame["sample_id"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions: {output_path}")
    if "is_at_risk" in frame.columns and frame["is_at_risk"].nunique() == 2:
        print(
            "AUROC on supplied labeled CSV: "
            f"{roc_auc_score(frame['is_at_risk'], probabilities):.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "predict"], default="search")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_DIR / "task1_optuna_cv_result.json"),
    )
    parser.add_argument("--output", default=str(PROJECT_DIR / "task1_optuna_predictions.csv"))
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
