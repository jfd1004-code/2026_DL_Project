"""
Patient-level image-domain audit for the Task 2 ultrasound dataset.
No model training and no GPU required.

Run from C:\\Users\\jfd10\\DL_Term:
python code\\vgg_patient_domain_audit.py ^
  --labels data\\T2_data\\labels.csv ^
  --predictions vgg_comparison\\vgg_comparison_predictions.csv ^
  --output-dir vgg_domain_audit
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


def resolve_path(value: str, labels_path: Path) -> Path:
    raw = str(value).strip().replace("\\", "/")
    rel = Path(raw)
    candidates = []
    if rel.is_absolute():
        candidates.append(rel)
    candidates.extend([
        labels_path.parent / rel,
        labels_path.parent / "images" / rel,
        labels_path.parent / "images" / rel.name,
        labels_path.parent.parent / rel,
        Path.cwd() / rel,
        Path.cwd() / "data" / "T2_data" / "images" / rel.name,
    ])
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"Image not found: {value}")


def inspect_image(path: Path):
    with Image.open(path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
        edge = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        prob = hist / hist.sum()
        nz = prob[prob > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        return {
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "aspect_ratio": float(arr.shape[1] / arr.shape[0]),
            "mean_gray": float(arr.mean()),
            "std_gray": float(arr.std()),
            "p05_gray": float(np.percentile(arr, 5)),
            "p50_gray": float(np.percentile(arr, 50)),
            "p95_gray": float(np.percentile(arr, 95)),
            "dark_frac_lt10": float((arr < 10).mean()),
            "dark_frac_lt25": float((arr < 25).mean()),
            "bright_frac_gt220": float((arr > 220).mean()),
            "edge_mean": float(edge.mean()),
            "entropy_bits": entropy,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output-dir", default="vgg_domain_audit")
    args = ap.parse_args()

    labels_path = Path(args.labels).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path, dtype={"sample_id": str, "patient_group": str})
    preds = pd.read_csv(predictions_path, dtype={"sample_id": str, "patient_group": str})
    need_l = {"sample_id", "path", "label", "patient_group"}
    need_p = {"sample_id", "patient_group", "label", "Optuna_VGG_probability"}
    if not need_l.issubset(labels.columns):
        raise ValueError(f"labels.csv needs columns: {sorted(need_l)}")
    if not need_p.issubset(preds.columns):
        raise ValueError(f"predictions CSV needs columns: {sorted(need_p)}")

    merged = labels.merge(
        preds[["sample_id", "patient_group", "label", "Optuna_VGG_probability"]],
        on=["sample_id", "patient_group", "label"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(preds):
        raise ValueError(f"Matched {len(merged)} of {len(preds)} predictions; dataset/split mismatch.")

    rows = []
    errors = []
    for i, row in merged.iterrows():
        try:
            path = resolve_path(row["path"], labels_path)
            feats = inspect_image(path)
            rows.append({
                "sample_id": row["sample_id"],
                "patient_group": row["patient_group"],
                "label": int(row["label"]),
                "optuna_probability": float(row["Optuna_VGG_probability"]),
                "image_path": str(path),
                **feats,
            })
        except Exception as e:
            errors.append({"sample_id": row["sample_id"], "error": str(e)})

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No images could be inspected.")
    detail_path = out_dir / "image_level_audit.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    feature_cols = [
        "width", "height", "aspect_ratio", "mean_gray", "std_gray",
        "p05_gray", "p50_gray", "p95_gray", "dark_frac_lt10",
        "dark_frac_lt25", "bright_frac_gt220", "edge_mean", "entropy_bits",
        "optuna_probability",
    ]
    summary = detail.groupby(["patient_group", "label"])[feature_cols].agg(
        ["count", "mean", "std", "min", "max"]
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary_path = out_dir / "patient_label_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    duplicates = detail.groupby("sha256").agg(
        count=("sample_id", "size"),
        sample_ids=("sample_id", lambda x: ";".join(map(str, x))),
        patients=("patient_group", lambda x: ";".join(sorted(set(map(str, x))))),
        labels=("label", lambda x: ";".join(map(str, x))),
    ).reset_index()
    duplicates = duplicates[duplicates["count"] > 1]
    duplicate_path = out_dir / "exact_duplicate_images.csv"
    duplicates.to_csv(duplicate_path, index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "image_read_errors.csv", index=False, encoding="utf-8-sig")

    print("=== PATIENT IMAGE DOMAIN AUDIT ===")
    print(f"Successfully inspected: {len(detail)} / {len(merged)}")
    print(f"Patients: {detail.patient_group.nunique()}")
    print(f"Exact duplicate groups: {len(duplicates)}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {duplicate_path}")
    if errors:
        print(f"Image read errors: {len(errors)} (see image_read_errors.csv)")
    print("\nPatient-level summary preview:")
    preview_cols = ["patient_group", "label", "mean_gray_mean", "std_gray_mean",
                    "dark_frac_lt25_mean", "edge_mean_mean", "entropy_bits_mean",
                    "optuna_probability_mean"]
    existing = [c for c in preview_cols if c in summary.columns]
    print(summary[existing].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
