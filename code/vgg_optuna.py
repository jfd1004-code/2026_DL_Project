"""
CPU-friendly Optuna hyperparameter search for the Task 2 VGG11 model.

Run from the project root:
    python code/vgg_optuna.py --csv data/T2_data/labels.csv

The script:
- Reuses the project's task2_split.csv (creates it with the same rule as vgg.py if absent).
- Keeps the official validation patients out of Optuna's search.
- Tunes parameters using 2-fold StratifiedGroupKFold on the official training patients.
- Uses BA at threshold 0.5 during Optuna search to avoid tuning a threshold in every trial.
- Trains a final VGG11 on the official training split and tunes one threshold on official validation.
- Saves vgg_optuna.pt, vgg_optuna.db, vgg_optuna_results.csv, and vgg_optuna_best_params.json.

No pretrained weights, ensembles, or external data are used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
SPLIT_PATH = ROOT / "task2_split.csv"
IMAGE_SIZE = (160, 112)  # H, W; retain the original VGG input size
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_project_path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        candidate = Path.cwd() / p
        if candidate.exists():
            return candidate.resolve()
        candidate = ROOT / p
        if candidate.exists():
            return candidate.resolve()
    return p.resolve()


def find_csv(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(resolve_project_path(value))
    else:
        candidates.extend([
            ROOT / "data" / "T2_data" / "labels.csv",
            ROOT / "T2_data" / "labels.csv",
            ROOT / "data" / "T2_data" / "T2_data" / "labels.csv",
        ])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "labels.csv not found. Example:\n"
        "python code/vgg_optuna.py --csv data/T2_data/labels.csv"
    )


def read_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sample_id": str, "patient_group": str})
    required = {"sample_id", "path", "label", "patient_group"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"labels.csv is missing columns: {missing}")
    if df[list(required)].isna().any().any():
        raise ValueError("labels.csv contains missing values in required columns.")
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["patient_group"] = df["patient_group"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")
    return df.reset_index(drop=True)


def resolve_image_path(value: str, csv_path: Path) -> Path:
    relative = Path(str(value).replace("\\", "/"))
    candidates = []
    if relative.is_absolute():
        candidates.append(relative)
    candidates.extend([
        csv_path.parent / relative,
        csv_path.parent / "images" / relative,
        csv_path.parent.parent / relative,
        ROOT / relative,
        ROOT / "data" / relative,
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Image not found: {value!r}. Checked paths relative to {csv_path.parent}"
    )


def load_or_create_official_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the split-generation rule in the original code/vgg.py.
    Existing task2_split.csv is never overwritten.
    """
    if SPLIT_PATH.is_file():
        split_df = pd.read_csv(SPLIT_PATH, dtype={"sample_id": str})
        if (
            not {"sample_id", "split"}.issubset(split_df.columns)
            or split_df["sample_id"].duplicated().any()
            or set(split_df["sample_id"]) != set(df["sample_id"])
        ):
            raise ValueError(
                "task2_split.csv is incompatible with labels.csv. "
                "Do not delete or replace it until the team has checked the shared split."
            )
        split_map = dict(zip(split_df["sample_id"], split_df["split"]))
        split_names = df["sample_id"].map(split_map)
        if split_names.isna().any() or not set(split_names.unique()).issubset({"train", "valid"}):
            raise ValueError("task2_split.csv must assign every sample to train or valid.")
        train_idx = np.flatnonzero(split_names.to_numpy() == "train")
        valid_idx = np.flatnonzero(split_names.to_numpy() == "valid")
        print(f"Reusing shared split: {SPLIT_PATH}")
    else:
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=SEED
        )
        candidates = []
        overall_positive_rate = float(df["label"].mean())
        for train_idx_candidate, valid_idx_candidate in splitter.split(
            df, df["label"], groups=df["patient_group"]
        ):
            train_labels = df.iloc[train_idx_candidate]["label"]
            valid_labels = df.iloc[valid_idx_candidate]["label"]
            if train_labels.nunique() != 2 or valid_labels.nunique() != 2:
                continue
            score = (
                abs(float(valid_labels.mean()) - overall_positive_rate)
                + 0.1 * abs(len(valid_idx_candidate) / len(df) - 0.2)
            )
            candidates.append((score, train_idx_candidate, valid_idx_candidate))
        if not candidates:
            raise ValueError("Could not create a patient-wise split containing both labels.")
        _, train_idx, valid_idx = min(candidates, key=lambda item: item[0])
        assignment = np.full(len(df), "train", dtype=object)
        assignment[valid_idx] = "valid"
        pd.DataFrame({
            "sample_id": df["sample_id"],
            "split": assignment,
        }).to_csv(SPLIT_PATH, index=False)
        print(f"Created shared split using the same rule as vgg.py: {SPLIT_PATH}")

    train_groups = set(df.iloc[train_idx]["patient_group"])
    valid_groups = set(df.iloc[valid_idx]["patient_group"])
    overlap = train_groups & valid_groups
    if overlap:
        raise ValueError(f"Patient leakage in task2_split.csv: {sorted(overlap)}")
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError("Train and validation splits must both be non-empty.")
    if df.iloc[train_idx]["label"].nunique() != 2 or df.iloc[valid_idx]["label"].nunique() != 2:
        raise ValueError("Train and validation must both contain labels 0 and 1.")

    print(
        f"Official split | Train: {len(train_idx)} images / {len(train_groups)} patients"
        f" | Valid: {len(valid_idx)} images / {len(valid_groups)} patients"
    )
    print(
        f"Positive counts | Train: {int(df.iloc[train_idx]['label'].sum())}"
        f" | Valid: {int(df.iloc[valid_idx]['label'].sum())}"
    )
    return train_idx, valid_idx


def split_sha256() -> str:
    return hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()


def make_transform(training: bool):
    ops = [transforms.Resize(IMAGE_SIZE)]
    if training:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(ops)


class UltrasoundDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, csv_path: Path, training: bool):
        self.paths = [
            resolve_image_path(value, csv_path)
            for value in frame["path"].astype(str).tolist()
        ]
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.transform = make_transform(training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            x = self.transform(image.convert("RGB"))
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


def build_model(dropout: float) -> nn.Module:
    # Same VGG11 architecture as the original project; no ImageNet weights.
    model = models.vgg11(weights=None, init_weights=False)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=float(dropout)),
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


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    outputs = []
    for images, _labels in loader:
        logits = model(images).view(-1)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def make_loader(
    frame: pd.DataFrame,
    csv_path: Path,
    training: bool,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = UltrasoundDataset(frame, csv_path, training=training)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,  # safest and usually efficient on Windows CPU
        pin_memory=False,
        generator=generator if training else None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_items += len(labels)
    return total_loss / max(1, total_items)


def find_inner_group_folds(train_df: pd.DataFrame, n_splits: int = 2):
    """
    Build inner folds only from the official training patients.
    The official validation patients are not used during Optuna search.
    """
    y = train_df["label"].to_numpy()
    groups = train_df["patient_group"].to_numpy()
    for random_state in range(SEED, SEED + 100):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        folds = list(splitter.split(train_df, y, groups=groups))
        valid = True
        for tr_idx, va_idx in folds:
            if (
                train_df.iloc[tr_idx]["label"].nunique() != 2
                or train_df.iloc[va_idx]["label"].nunique() != 2
            ):
                valid = False
                break
            tr_groups = set(train_df.iloc[tr_idx]["patient_group"])
            va_groups = set(train_df.iloc[va_idx]["patient_group"])
            if tr_groups & va_groups:
                valid = False
                break
        if valid:
            return folds, random_state
    raise ValueError(
        "Could not construct 2 patient-group CV folds with both classes. "
        "Check patient_group and label columns."
    )


def objective_factory(
    train_df: pd.DataFrame,
    csv_path: Path,
    inner_folds,
    trial_epochs: int,
    threads: int,
):
    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 2e-5, 2e-4, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-5, 1e-2, log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.20, 0.60),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "pos_weight_multiplier": trial.suggest_float(
                "pos_weight_multiplier", 0.50, 1.20
            ),
        }

        fold_scores = []
        fold_best_epochs = []

        for fold_number, (inner_train_idx, inner_valid_idx) in enumerate(
            inner_folds, start=1
        ):
            # Same seed per fold across trials makes the comparison more controlled.
            set_seed(SEED + fold_number)
            torch.set_num_threads(threads)

            fold_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
            fold_valid_df = train_df.iloc[inner_valid_idx].reset_index(drop=True)

            train_loader = make_loader(
                fold_train_df, csv_path, True, params["batch_size"],
                seed=SEED + fold_number,
            )
            valid_loader = make_loader(
                fold_valid_df, csv_path, False, params["batch_size"],
                seed=SEED + fold_number,
            )

            n_positive = int(fold_train_df["label"].sum())
            n_negative = len(fold_train_df) - n_positive
            base_pos_weight = n_negative / max(1, n_positive)
            pos_weight = base_pos_weight * params["pos_weight_multiplier"]

            model = build_model(params["dropout"])
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
            )
            y_valid = fold_valid_df["label"].to_numpy(dtype=int)
            best_fold_ba = -1.0
            best_fold_epoch = 1

            for epoch in range(1, trial_epochs + 1):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, loss_fn
                )
                probabilities = predict_probabilities(model, valid_loader)
                ba = float(
                    balanced_accuracy_score(
                        y_valid, (probabilities >= 0.5).astype(int)
                    )
                )
                scheduler.step(ba)

                if ba > best_fold_ba + 1e-6:
                    best_fold_ba = ba
                    best_fold_epoch = epoch

                print(
                    f"  Trial {trial.number + 1} | Fold {fold_number}/2"
                    f" | Epoch {epoch}/{trial_epochs}"
                    f" | Loss {train_loss:.4f} | BA@0.5 {ba:.4f}",
                    flush=True,
                )

            fold_scores.append(best_fold_ba)
            fold_best_epochs.append(best_fold_epoch)

            del model, optimizer, scheduler, loss_fn, train_loader, valid_loader
            gc.collect()

        mean_cv_ba = float(np.mean(fold_scores))
        trial.set_user_attr("fold_best_ba", fold_scores)
        trial.set_user_attr("fold_best_epochs", fold_best_epochs)
        print(
            f"[Trial {trial.number + 1}] Mean 2-fold CV BA@0.5: "
            f"{mean_cv_ba:.4f} | fold scores={fold_scores} | params={params}",
            flush=True,
        )
        return mean_cv_ba

    return objective


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray):
    candidates = np.linspace(0.10, 0.90, 33)
    candidates = sorted(candidates, key=lambda value: abs(value - 0.5))
    best_ba = -1.0
    best_threshold = 0.5
    for threshold in candidates:
        ba = float(
            balanced_accuracy_score(
                y_true, (probabilities >= threshold).astype(int)
            )
        )
        if ba > best_ba + 1e-9:
            best_ba = ba
            best_threshold = float(threshold)
    return best_threshold, best_ba


def final_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    csv_path: Path,
    params: dict,
    output_path: Path,
    max_epochs: int,
    patience: int,
    threads: int,
):
    set_seed(SEED)
    torch.set_num_threads(threads)

    batch_size = int(params["batch_size"])
    train_loader = make_loader(
        train_df, csv_path, True, batch_size, seed=SEED
    )
    valid_loader = make_loader(
        valid_df, csv_path, False, batch_size, seed=SEED
    )

    n_positive = int(train_df["label"].sum())
    n_negative = len(train_df) - n_positive
    base_pos_weight = n_negative / max(1, n_positive)
    pos_weight = base_pos_weight * float(params["pos_weight_multiplier"])

    model = build_model(float(params["dropout"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    y_valid = valid_df["label"].to_numpy(dtype=int)
    best_ba_at_half = -1.0
    best_epoch = 1
    best_state = None
    wait = 0

    print("\n=== Final VGG fit on the official training split ===", flush=True)
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        probabilities = predict_probabilities(model, valid_loader)
        ba_at_half = float(
            balanced_accuracy_score(
                y_valid, (probabilities >= 0.5).astype(int)
            )
        )
        scheduler.step(ba_at_half)
        print(
            f"Final Epoch {epoch:02d}/{max_epochs}"
            f" | Loss {train_loss:.4f}"
            f" | Validation BA@0.5 {ba_at_half:.4f}"
            f" | LR {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        if ba_at_half > best_ba_at_half + 1e-6:
            best_ba_at_half = ba_at_half
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Final fit early-stopped after {patience} epochs without improvement.")
                break

    if best_state is None:
        raise RuntimeError("Final fit did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    probabilities = predict_probabilities(model, valid_loader)
    threshold, tuned_ba = choose_threshold(y_valid, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_valid, predictions, labels=[0, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "model_name": "vgg11",
        "image_size": list(IMAGE_SIZE),
        "threshold": float(threshold),
        "best_valid_ba": float(tuned_ba),
        "best_valid_ba_tuned": float(tuned_ba),
        "best_valid_ba_at_05": float(best_ba_at_half),
        "best_epoch": int(best_epoch),
        "mean": list(MEAN),
        "std": list(STD),
        "optuna_best_params": params,
        "split_file": str(SPLIT_PATH.name),
        "split_sha256": split_sha256(),
        "validation_threshold_selected_on_validation": True,
    }
    torch.save(checkpoint, output_path)

    print("\n=== Optuna VGG Result ===")
    print(f"Best epoch by Validation BA@0.5: {best_epoch}")
    print(f"Validation BA@0.5: {best_ba_at_half:.4f}")
    print(f"Validation BA after threshold selection: {tuned_ba:.4f}")
    print(f"Selected threshold: {threshold:.3f}")
    print(f"Confusion matrix [true rows 0/1, predicted cols 0/1]:\n{cm}")
    print(f"Saved weights: {output_path}")
    print("Checkpoint load test: ", end="", flush=True)

    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    verify_model = build_model(float(params["dropout"]))
    verify_model.load_state_dict(loaded["model_state_dict"])
    verify_model.eval()
    print("OK")
    print(
        "Note: threshold was selected on the official validation split; "
        "its reported BA is a model-selection estimate, not private-test performance."
    )

    del model, verify_model, optimizer, scheduler, loss_fn
    del train_loader, valid_loader
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="CPU-friendly Optuna hyperparameter optimization for VGG11."
    )
    parser.add_argument("--csv", default=None, help="Path to T2 labels.csv")
    parser.add_argument("--trials", type=int, default=6,
                        help="Number of additional Optuna trials (default: 6)")
    parser.add_argument("--trial-epochs", type=int, default=2,
                        help="Maximum epochs per inner-CV fold and trial (default: 2)")
    parser.add_argument("--final-epochs", type=int, default=6,
                        help="Maximum epochs for final official train/valid fit (default: 6)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience for final fit")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for PyTorch (default: 4)")
    parser.add_argument("--study-name", default="vgg11_cpu_optuna")
    parser.add_argument("--output", default="vgg_optuna.pt",
                        help="Output checkpoint path, relative to project root by default")
    args = parser.parse_args()

    if args.trials < 1 or args.trial_epochs < 1 or args.final_epochs < 1:
        raise ValueError("--trials, --trial-epochs and --final-epochs must be >= 1")

    try:
        import optuna
    except ImportError:
        raise SystemExit(
            "Optuna is not installed. Install it with:\n"
            "python -m pip install optuna"
        )

    threads = max(1, int(args.threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    csv_path = find_csv(args.csv)
    df = read_labels(csv_path)
    train_idx, valid_idx = load_or_create_official_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    inner_folds, inner_seed = find_inner_group_folds(train_df, n_splits=2)
    inner_fold_patients = []
    for fold_number, (inner_train_idx, inner_valid_idx) in enumerate(inner_folds, start=1):
        tr_groups = sorted(train_df.iloc[inner_train_idx]["patient_group"].unique().tolist())
        va_groups = sorted(train_df.iloc[inner_valid_idx]["patient_group"].unique().tolist())
        inner_fold_patients.append({
            "fold": fold_number,
            "train_patients": tr_groups,
            "valid_patients": va_groups,
        })

    print("\n=== CPU Optuna VGG11 Search ===")
    print(f"Device: CPU | Threads: {threads}")
    print(f"CSV: {csv_path}")
    print(f"Images: {len(df)} | Patients: {df['patient_group'].nunique()}")
    print(f"Official train/valid split SHA256: {split_sha256()}")
    print(f"Optuna inner CV: 2 patient-group folds (random_state={inner_seed})")
    print(f"Trials to add: {args.trials} | Epochs per fold/trial: {args.trial_epochs}")
    print("Search metric: mean fold-best Balanced Accuracy at threshold 0.5")
    print("The official validation patients are not used in Optuna trials.\n")

    db_path = ROOT / "vgg_optuna.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    objective = objective_factory(
        train_df, csv_path, inner_folds, args.trial_epochs, threads,
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not completed:
        raise RuntimeError(
            "No Optuna trial completed successfully. Check the error messages above."
        )

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    print("\n=== Optuna Search Summary ===")
    print(f"Completed trials: {len(completed)} / total stored: {len(study.trials)}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best 2-fold CV BA@0.5: {best_trial.value:.4f}")
    print(f"Best parameters: {best_params}")

    results_path = ROOT / "vgg_optuna_results.csv"
    study.trials_dataframe(
        attrs=("number", "value", "params", "state", "duration")
    ).to_csv(results_path, index=False)

    params_path = ROOT / "vgg_optuna_best_params.json"
    params_payload = {
        "study_name": args.study_name,
        "best_trial_number": int(best_trial.number),
        "best_inner_cv_ba_at_05": float(best_trial.value),
        "best_params": best_params,
        "official_split_sha256": split_sha256(),
        "inner_cv_random_state": int(inner_seed),
        "inner_cv_patient_groups": inner_fold_patients,
        "note": (
            "Optuna objective is the mean of each inner patient-group fold's "
            "best Balanced Accuracy at threshold 0.5. Official validation patients "
            "were not used during hyperparameter search."
        ),
    }
    params_path.write_text(
        json.dumps(params_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved trial table: {results_path}")
    print(f"Saved best parameters: {params_path}")
    print(f"Saved Optuna database: {db_path}")

    output_path = resolve_project_path(args.output)
    final_train(
        train_df=train_df,
        valid_df=valid_df,
        csv_path=csv_path,
        params=best_params,
        output_path=output_path,
        max_epochs=args.final_epochs,
        patience=args.patience,
        threads=threads,
    )


if __name__ == "__main__":
    main()
