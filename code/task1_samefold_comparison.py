"""
Compare the original Task 1 DNN and the Optuna-selected DNN
under exactly the same 3-fold stratified CV protocol.

Run from project root:
python code/task1_samefold_comparison.py --csv data/T1_data/heloc_release.csv --result-json task1_optuna_cv_result.json --device cpu

This script:
- Does NOT run Optuna or modify either checkpoint.
- Uses StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).
- Uses the same fold indices and per-fold random seed for both configurations.
- Fits preprocessing separately inside each training fold.
- Saves fold-level and summary results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# This file must be placed in the project's code/ directory beside
# task1_optuna_cv.py.
import task1_optuna_cv as cv


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same-fold comparison: baseline DNN vs Optuna DNN"
    )
    parser.add_argument(
        "--csv",
        default="data/T1_data/heloc_release.csv",
        help="Labeled Task 1 training CSV",
    )
    parser.add_argument(
        "--result-json",
        default="task1_optuna_cv_result.json",
        help="JSON produced by task1_optuna_cv.py",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--output-prefix",
        default="task1_samefold_comparison",
        help="Output file prefix, relative to project root unless absolute",
    )
    args = parser.parse_args()

    csv_path = resolve_path(args.csv)
    result_path = resolve_path(args.result_json)
    output_prefix = resolve_path(args.output_prefix)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not result_path.is_file():
        raise FileNotFoundError(f"Optuna result JSON not found: {result_path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    optuna_params = result.get("best_params")
    if not isinstance(optuna_params, dict) or "hidden_dims" not in optuna_params:
        raise ValueError(
            "The result JSON must contain best_params in the converted format "
            "with a hidden_dims field."
        )

    # Convert the baseline's Optuna-search representation into the same
    # parameter format consumed by cv.train_fold().
    baseline_params = cv.convert_trial_params(cv.params_for_baseline())

    df, feature_columns, y = cv.load_data(csv_path)
    df = df.copy()
    df["_target"] = y

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    folds = list(splitter.split(df, y))

    configurations = [
        ("Baseline_DNN", baseline_params),
        ("Optuna_DNN", optuna_params),
    ]

    print("=== TASK 1 SAME-FOLD COMPARISON ===", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Rows: {len(df)} | Features: {len(feature_columns)}", flush=True)
    print(f"Device: {device}", flush=True)
    print("CV: StratifiedKFold(3), shuffle=True, random_state=2026", flush=True)
    print(f"Training: max_epochs={args.max_epochs}, patience={args.patience}", flush=True)
    print("Both models use identical fold indices and fold seeds.", flush=True)
    print("Preprocessing is fitted on each fold's training rows only.", flush=True)
    print(f"Baseline config: {baseline_params}", flush=True)
    print(f"Optuna config: {optuna_params}", flush=True)

    rows: list[dict] = []
    scores: dict[str, list[float]] = {
        name: [] for name, _ in configurations
    }

    for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        fold_seed = 2026 + fold_no

        print(
            f"\n--- Fold {fold_no}/3 | train={len(train_df)} "
            f"valid={len(valid_df)} | seed={fold_seed} ---",
            flush=True,
        )

        for name, params in configurations:
            auc, best_epoch = cv.train_fold(
                train_df=train_df,
                valid_df=valid_df,
                feature_columns=feature_columns,
                params=params,
                seed=fold_seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
            )
            scores[name].append(float(auc))
            rows.append(
                {
                    "model": name,
                    "fold": fold_no,
                    "seed": fold_seed,
                    "train_rows": len(train_df),
                    "validation_rows": len(valid_df),
                    "best_epoch": int(best_epoch),
                    "validation_auroc": float(auc),
                }
            )
            print(
                f"{name}: AUROC={auc:.6f} | best_epoch={best_epoch}",
                flush=True,
            )

    baseline_scores = np.asarray(scores["Baseline_DNN"], dtype=float)
    optuna_scores = np.asarray(scores["Optuna_DNN"], dtype=float)
    differences = optuna_scores - baseline_scores

    summary = {
        "csv": str(csv_path),
        "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)",
        "device": str(device),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "fold_seeds": [2027, 2028, 2029],
        "baseline_params": baseline_params,
        "optuna_params": optuna_params,
        "baseline_fold_auroc": baseline_scores.tolist(),
        "optuna_fold_auroc": optuna_scores.tolist(),
        "paired_fold_difference_optuna_minus_baseline": differences.tolist(),
        "baseline_mean_auroc": float(baseline_scores.mean()),
        "baseline_std_auroc_population": float(baseline_scores.std(ddof=0)),
        "optuna_mean_auroc": float(optuna_scores.mean()),
        "optuna_std_auroc_population": float(optuna_scores.std(ddof=0)),
        "mean_paired_difference": float(differences.mean()),
        "optuna_wins_folds": int((differences > 0).sum()),
        "baseline_wins_folds": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "interpretation_note": (
            "This is a controlled same-fold comparison, but the Optuna "
            "configuration was selected using this same CV protocol. Its "
            "score therefore remains subject to hyperparameter-selection "
            "optimism and is not an independent estimate of private-test AUROC. "
            "Do not describe this as private-test performance."
        ),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fold_csv = output_prefix.with_name(output_prefix.name + "_folds.csv")
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")

    pd.DataFrame(rows).to_csv(fold_csv, index=False)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== COMPARISON SUMMARY ===", flush=True)
    print(
        "Baseline DNN: "
        f"{baseline_scores.mean():.5f} ± {baseline_scores.std(ddof=0):.5f}",
        flush=True,
    )
    print(
        "Optuna DNN:   "
        f"{optuna_scores.mean():.5f} ± {optuna_scores.std(ddof=0):.5f}",
        flush=True,
    )
    print(
        "Optuna - Baseline fold differences: "
        f"{differences.tolist()}",
        flush=True,
    )
    print(f"Optuna wins: {(differences > 0).sum()}/3 folds", flush=True)
    print(f"Saved fold results: {fold_csv}", flush=True)
    print(f"Saved summary: {summary_json}", flush=True)
    print("No model checkpoints were trained, overwritten, or modified.", flush=True)


if __name__ == "__main__":
    main()
