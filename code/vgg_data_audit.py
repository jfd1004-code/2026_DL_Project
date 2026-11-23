from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import statistics
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def resolve_image_path(value: str, csv_path: Path, project_root: Path) -> Path | None:
    rel = Path(str(value).replace('\\', '/'))
    candidates = []
    if rel.is_absolute():
        candidates.append(rel)
    candidates += [
        csv_path.parent / rel,
        csv_path.parent / 'images' / rel,
        csv_path.parent.parent / rel,
        project_root / rel,
        project_root / 'data' / rel,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Audit Task 2 VGG dataset locally; no image upload required.')
    parser.add_argument('--csv', default='data/T2_data/labels.csv')
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'labels.csv not found: {csv_path}')

    df = pd.read_csv(csv_path, dtype={'sample_id': str, 'patient_group': str})
    required = {'sample_id', 'path', 'label', 'patient_group'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}; found: {list(df.columns)}')
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)
    if not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError('label must contain only 0/1')

    lines = []
    def out(s=''):
        print(s)
        lines.append(str(s))

    out('=== TASK 2 DATA AUDIT ===')
    out(f'CSV: {csv_path}')
    out(f'Rows: {len(df)}')
    out(f'Columns: {list(df.columns)}')
    out(f'Duplicate sample_id: {int(df.sample_id.duplicated().sum())}')
    out(f'Missing values by required column: {df[list(required)].isna().sum().to_dict()}')
    out(f'Label counts: {df.label.value_counts().sort_index().to_dict()}')
    out(f'Positive rate: {df.label.mean():.4f}')
    out(f'Patients: {df.patient_group.nunique()}')
    per_patient = df.groupby('patient_group').agg(images=('sample_id', 'size'), positives=('label', 'sum'))
    out(f'Images per patient min/median/max: {per_patient.images.min()}/{per_patient.images.median():.1f}/{per_patient.images.max()}')
    out('Per-patient counts (patient_group, images, positives):')
    for patient, row in per_patient.sort_index().iterrows():
        out(f'  {patient}: {int(row.images)} images, {int(row.positives)} positive')

    found, missing_paths, bad_images = [], [], []
    dims, modes = Counter(), Counter()
    for value in df.path.astype(str):
        p = resolve_image_path(value, csv_path, project_root)
        if p is None:
            missing_paths.append(value)
            continue
        found.append(p)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                dims[im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            bad_images.append((value, str(e)))
    out(f'Image paths found: {len(found)}/{len(df)}')
    out(f'Missing image paths: {len(missing_paths)}')
    for p in missing_paths[:10]:
        out(f'  MISSING: {p}')
    out(f'Unreadable/corrupt images: {len(bad_images)}')
    for p, e in bad_images[:5]:
        out(f'  BAD: {p}: {e}')
    out(f'Image dimensions (width, height): {dict(dims)}')
    out(f'Image modes: {dict(modes)}')

    out('=== PATIENT-GROUP SPLIT CHECK ===')
    if df.patient_group.nunique() >= 5:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        candidates = []
        overall_rate = df.label.mean()
        for fold, (tr, va) in enumerate(sgkf.split(df, df.label, groups=df.patient_group), 1):
            tr_groups = set(df.iloc[tr].patient_group)
            va_groups = set(df.iloc[va].patient_group)
            overlap = len(tr_groups & va_groups)
            valid_two_classes = df.iloc[tr].label.nunique() == 2 and df.iloc[va].label.nunique() == 2
            score = abs(df.iloc[va].label.mean() - overall_rate) + 0.1 * abs(len(va)/len(df)-0.2)
            out(f'Fold {fold}: train={len(tr)} images/{len(tr_groups)} patients, valid={len(va)} images/{len(va_groups)} patients, valid_labels={df.iloc[va].label.value_counts().sort_index().to_dict()}, overlap={overlap}, usable={valid_two_classes}')
            if valid_two_classes and overlap == 0:
                candidates.append((score, fold))
        if candidates:
            out(f'Best-balanced candidate fold by current script rule: {min(candidates)[1]}')
    else:
        out('Fewer than 5 patients; 5-fold StratifiedGroupKFold cannot be used safely.')

    split_path = project_root / 'task2_split.csv'
    if split_path.is_file():
        split_df = pd.read_csv(split_path, dtype={'sample_id': str})
        out(f'Existing split file: {split_path}')
        if {'sample_id', 'split'}.issubset(split_df.columns) and not split_df.sample_id.duplicated().any() and set(split_df.sample_id) == set(df.sample_id):
            merged = df[['sample_id', 'patient_group', 'label']].merge(split_df, on='sample_id', how='left')
            tr_g = set(merged.loc[merged.split == 'train', 'patient_group'])
            va_g = set(merged.loc[merged.split == 'valid', 'patient_group'])
            out(f'Existing split labels: {merged.groupby("split").label.value_counts().to_dict()}')
            out(f'Existing split patient overlap: {len(tr_g & va_g)}')
        else:
            out('Existing split file is incompatible with labels.csv.')
    else:
        out('No existing task2_split.csv; current vgg.py will create one on first training run.')

    report = project_root / 'vgg_data_audit.txt'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out(f'Report saved: {report}')

if __name__ == '__main__':
    main()
