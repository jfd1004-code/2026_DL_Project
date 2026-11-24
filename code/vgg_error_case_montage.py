"""
Export hard validation examples as contact sheets for visual error analysis.

Run from C:\\Users\\jfd10\\DL_Term:
python code\\vgg_error_case_montage.py ^
  --labels data\\T2_data\\labels.csv ^
  --predictions vgg_comparison\\vgg_comparison_predictions.csv ^
  --output-dir vgg_error_montages

For each patient_group, exports the hardest negative examples (label 0 with
highest Optuna probability) and hardest positive examples (label 1 with
lowest Optuna probability). These are the most useful images to inspect
because they are closest to / on the wrong side of the model decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def resolve_image_path(value: str, labels_path: Path) -> Path:
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
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found for path={value!r}")


def load_font(size=15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def make_tile(image_path, title, subtitle, tile_w=260, image_h=190):
    with Image.open(image_path) as src:
        img = src.convert("RGB")
        img.thumbnail((tile_w - 12, image_h - 12), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (tile_w, image_h + 62), "white")
    x = (tile_w - img.width) // 2
    y = 6 + (image_h - 12 - img.height) // 2
    tile.paste(img, (x, y))

    draw = ImageDraw.Draw(tile)
    font = load_font(14)
    small = load_font(12)
    draw.text((7, image_h + 4), title[:34], fill="black", font=font)
    draw.text((7, image_h + 27), subtitle[:42], fill="black", font=small)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output-dir", default="vgg_error_montages")
    ap.add_argument("--hard-cases-per-class", type=int, default=8)
    args = ap.parse_args()

    labels_path = Path(args.labels).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path, dtype={"sample_id": str, "patient_group": str})
    preds = pd.read_csv(predictions_path, dtype={"sample_id": str, "patient_group": str})

    required_labels = {"sample_id", "path", "label", "patient_group"}
    required_preds = {
        "sample_id", "patient_group", "label", "Optuna_VGG_probability",
        "Baseline_VGG_probability"
    }
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"labels CSV must include {sorted(required_labels)}")
    if not required_preds.issubset(preds.columns):
        raise ValueError(f"prediction CSV must include {sorted(required_preds)}")

    merged = labels.merge(
        preds[["sample_id", "patient_group", "label",
               "Optuna_VGG_probability", "Baseline_VGG_probability"]],
        on=["sample_id", "patient_group", "label"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(preds):
        raise ValueError(
            f"Only matched {len(merged)} of {len(preds)} prediction rows. "
            "Check that labels.csv and prediction CSV are from the same split/dataset."
        )

    manifest_rows = []
    for patient, group in merged.groupby("patient_group", sort=True):
        # Most suspicious examples: high score for absent, low score for present.
        neg = group[group["label"] == 0].sort_values(
            "Optuna_VGG_probability", ascending=False
        ).head(args.hard_cases_per_class)
        pos = group[group["label"] == 1].sort_values(
            "Optuna_VGG_probability", ascending=True
        ).head(args.hard_cases_per_class)

        selected = pd.concat([neg, pos], axis=0)
        tiles = []
        for _, row in selected.iterrows():
            try:
                img_path = resolve_image_path(row["path"], labels_path)
                y = int(row["label"])
                opt_p = float(row["Optuna_VGG_probability"])
                base_p = float(row["Baseline_VGG_probability"])
                kind = "ABSENT / hard FP" if y == 0 else "PRESENT / hard FN"
                title = f"{kind} | true={y}"
                subtitle = (
                    f"Optuna={opt_p:.4f}  Baseline={base_p:.4f}  "
                    f"ID={str(row['sample_id'])[:8]}"
                )
                tiles.append(make_tile(img_path, title, subtitle))
                manifest_rows.append({
                    "patient_group": patient,
                    "sample_id": row["sample_id"],
                    "label": y,
                    "path": str(img_path),
                    "optuna_probability": opt_p,
                    "baseline_probability": base_p,
                    "case_type": kind,
                })
            except Exception as exc:
                manifest_rows.append({
                    "patient_group": patient,
                    "sample_id": row["sample_id"],
                    "label": int(row["label"]),
                    "path": str(row["path"]),
                    "optuna_probability": float(row["Optuna_VGG_probability"]),
                    "baseline_probability": float(row["Baseline_VGG_probability"]),
                    "case_type": f"IMAGE_ERROR: {exc}",
                })

        if not tiles:
            print(f"No readable images for patient {patient}; skipped.")
            continue

        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        tile_w, tile_h = tiles[0].size
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (225, 225, 225))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
        out_path = output_dir / f"patient_{patient}_hard_cases.jpg"
        sheet.save(out_path, quality=92)
        print(f"Saved {out_path} ({len(tiles)} images)")

    manifest_path = output_dir / "selected_cases_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved manifest: {manifest_path}")
    print("Upload the generated patient_*.jpg contact sheets for visual inspection.")


if __name__ == "__main__":
    main()
