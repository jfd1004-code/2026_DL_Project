"""
Task 2 — Train VGG11, ResNet18, and GoogLeNet separately with structured regularization.

Important compliance choices:
- Exactly one VGG, one ResNet, and one GoogLeNet are trained independently.
- Each model is trained and evaluated separately. No prediction averaging or ensemble.
- No pretrained weights and no external data.
- The same existing patient-wise task2_split.csv is required and reused.
- DropBlock is used as a CNN regularizer (structured dropout).
- Stochastic Depth is implemented for ResNet but defaults to OFF because the
  assignment bans ensembles and Stochastic Depth is sometimes described as an
  implicit ensemble. Enable it only if the instructor confirms this interpretation.

Run from project root:
python code\\task2_train_structured.py --csv data\\T2_data\\labels.csv --split-csv task2_split.csv --output-dir task2_structured_runs --device auto --models all

Recommended conservative run (default):
- VGG11 + DropBlock
- ResNet18 without Stochastic Depth (single ordinary ResNet18)
- GoogLeNet + DropBlock

If Stochastic Depth is confirmed as acceptable:
python code\\task2_train_structured.py --csv data\\T2_data\\labels.csv --split-csv task2_split.csv --output-dir task2_structured_runs_sd --device auto --models resnet --stochastic-depth 0.10

The script trains sequentially, saves a resumable checkpoint every epoch, saves
the best checkpoint, writes logs/metrics, and continues to the next architecture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "T2_data" / "labels.csv"
DEFAULT_SPLIT = ROOT / "task2_split.csv"
DEFAULT_OUTPUT_DIR = ROOT / "task2_structured_runs"

SEED = 42
IMAGE_SIZE = (160, 112)  # H, W — same input dimensions used by previous Task 2 VGG runs
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

MODEL_ORDER = ["vgg", "resnet", "googlenet"]
MODEL_FILES = {
    "vgg": "vgg11_dropblock.pt",
    "resnet": "resnet18.pt",
    "googlenet": "googlenet_dropblock.pt",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("task2_structured")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        output_dir / "task2_training.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def read_labels(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {csv_path}")
    df = pd.read_csv(
        csv_path,
        dtype={"sample_id": str, "patient_group": str},
    )
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
    raw = str(value).strip().replace("\\", "/")
    relative = Path(raw)
    candidates: list[Path] = []
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
        f"Image not found for path={value!r}; checked paths relative to {csv_path.parent}"
    )


def load_shared_split(
    df: pd.DataFrame,
    split_path: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    # Never generate a new split here. All three architectures must use the
    # same existing patient-wise split already used by the project.
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Required shared split file not found: {split_path}\n"
            "Copy the team's existing task2_split.csv. This script will not "
            "create or replace it."
        )
    split_df = pd.read_csv(split_path, dtype={"sample_id": str})
    if not {"sample_id", "split"}.issubset(split_df.columns):
        raise ValueError("task2_split.csv must contain sample_id and split.")
    if split_df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in task2_split.csv.")
    if set(split_df["sample_id"].astype(str)) != set(df["sample_id"].astype(str)):
        raise ValueError(
            "task2_split.csv sample IDs do not exactly match labels.csv. "
            "Do not overwrite the shared split."
        )

    mapping = dict(zip(
        split_df["sample_id"].astype(str),
        split_df["split"].astype(str),
    ))
    split_values = df["sample_id"].map(mapping)
    if split_values.isna().any():
        raise ValueError("Some rows have no split assignment.")
    if not set(split_values.unique()).issubset({"train", "valid"}):
        raise ValueError("Split values must be exactly 'train' and 'valid'.")

    train_idx = np.flatnonzero(split_values.to_numpy() == "train")
    valid_idx = np.flatnonzero(split_values.to_numpy() == "valid")
    train_groups = set(df.iloc[train_idx]["patient_group"])
    valid_groups = set(df.iloc[valid_idx]["patient_group"])
    overlap = train_groups & valid_groups
    if overlap:
        raise ValueError(f"Patient leakage detected: {sorted(overlap)}")
    if not len(train_idx) or not len(valid_idx):
        raise ValueError("Train and validation splits must be non-empty.")
    if df.iloc[train_idx]["label"].nunique() != 2:
        raise ValueError("Train split must contain both labels.")
    if df.iloc[valid_idx]["label"].nunique() != 2:
        raise ValueError("Validation split must contain both labels.")

    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    return train_idx, valid_idx, split_hash


class UltrasoundDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        csv_path: Path,
        training: bool,
    ):
        self.paths = [
            resolve_image_path(p, csv_path)
            for p in frame["path"].astype(str)
        ]
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.sample_ids = frame["sample_id"].astype(str).to_numpy()
        self.patient_groups = frame["patient_group"].astype(str).to_numpy()

        ops: list[Any] = [transforms.Resize(IMAGE_SIZE)]
        if training:
            ops.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image_file:
            image = self.transform(image_file.convert("RGB"))
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label


class DropBlock2D(nn.Module):
    """
    Structured dropout for CNN feature maps.
    Randomly removes contiguous spatial blocks during training only.
    It is not a separately trained model and is disabled in evaluation mode.
    """
    def __init__(self, drop_prob: float = 0.08, block_size: int = 3):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        if x.ndim != 4:
            return x

        batch, channels, height, width = x.shape
        block = min(self.block_size, height, width)
        if block < 2:
            return x

        gamma = (
            self.drop_prob * height * width
            / (block * block * max(1, height - block + 1) * max(1, width - block + 1))
        )
        gamma = min(max(gamma, 0.0), 1.0)

        # A separate mask per sample and channel.
        seeds = (torch.rand(
            (batch, channels, height, width),
            device=x.device,
            dtype=torch.float32,
        ) < gamma).float()

        block_mask = F.max_pool2d(
            seeds,
            kernel_size=block,
            stride=1,
            padding=block // 2,
        )
        block_mask = block_mask[:, :, :height, :width]
        keep_mask = 1.0 - block_mask

        # Preserve the expected activation magnitude.
        keep_count = keep_mask.sum().clamp_min(1.0)
        scale = keep_mask.numel() / keep_count
        return x * keep_mask.to(dtype=x.dtype) * scale.to(dtype=x.dtype)


class DropBlockWrapper(nn.Module):
    def __init__(self, module: nn.Module, drop_prob: float, block_size: int = 3):
        super().__init__()
        self.module = module
        self.dropblock = DropBlock2D(drop_prob, block_size)

    def forward(self, x):
        y = self.module(x)
        if isinstance(y, torch.Tensor) and y.ndim == 4:
            y = self.dropblock(y)
        return y


class StochasticBasicBlock(models.resnet.BasicBlock):
    """
    ResNet BasicBlock with optional stochastic depth on the residual branch.
    drop_prob=0 exactly recovers the ordinary BasicBlock forward path.
    """
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ):
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        self.drop_prob = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            mask = torch.empty(
                (out.shape[0], 1, 1, 1),
                dtype=out.dtype,
                device=out.device,
            ).bernoulli_(keep_prob)
            out = out * mask / keep_prob

        out = out + identity
        out = self.relu(out)
        return out


def build_vgg11(dropblock_prob: float) -> nn.Module:
    model = models.vgg11(weights=None, init_weights=False)
    new_features: list[nn.Module] = []
    for layer in model.features:
        new_features.append(layer)
        if isinstance(layer, nn.MaxPool2d) and dropblock_prob > 0:
            new_features.append(DropBlock2D(dropblock_prob, block_size=3))
    model.features = nn.Sequential(*new_features)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
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


def build_resnet18(stochastic_depth_prob: float, dropblock_prob: float) -> nn.Module:
    model = models.resnet.ResNet(
        block=StochasticBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1,
    )
    blocks = [
        module for module in model.modules()
        if isinstance(module, StochasticBasicBlock)
    ]
    if stochastic_depth_prob > 0 and blocks:
        for index, block in enumerate(blocks):
            # Linear drop-probability schedule: shallow blocks are dropped
            # less often than deeper blocks.
            block.drop_prob = (
                float(stochastic_depth_prob) * (index + 1) / len(blocks)
            )
    # DropBlock is applied to stage feature maps; the residual architecture
    # itself remains a single ordinary ResNet18 when stochastic_depth_prob=0.
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(model, stage_name)
        setattr(
            model,
            stage_name,
            DropBlockWrapper(stage, dropblock_prob, block_size=3)
            if dropblock_prob > 0 else stage,
        )
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


def build_googlenet(dropblock_prob: float) -> nn.Module:
    model = models.googlenet(
        weights=None,
        aux_logits=False,
        num_classes=1,
        dropout=0.35,
        init_weights=False,
        transform_input=False,
    )
    if dropblock_prob > 0:
        inception_names = [
            "inception3a", "inception3b",
            "inception4a", "inception4b", "inception4c",
            "inception4d", "inception4e",
            "inception5a", "inception5b",
        ]
        for name in inception_names:
            module = getattr(model, name)
            setattr(
                model,
                name,
                DropBlockWrapper(module, dropblock_prob, block_size=3),
            )
    return model


def build_model(
    name: str,
    dropblock_prob: float,
    stochastic_depth_prob: float,
) -> nn.Module:
    if name == "vgg":
        return build_vgg11(dropblock_prob)
    if name == "resnet":
        return build_resnet18(stochastic_depth_prob, dropblock_prob)
    if name == "googlenet":
        return build_googlenet(dropblock_prob)
    raise ValueError(f"Unknown model name: {name}")


def model_config(name: str, dropblock_prob: float, stochastic_depth_prob: float):
    return {
        "model_name": name,
        "architecture": {
            "vgg": "VGG11",
            "resnet": "ResNet18",
            "googlenet": "GoogLeNet",
        }[name],
        "dropblock_prob": float(dropblock_prob),
        "stochastic_depth_prob": float(stochastic_depth_prob if name == "resnet" else 0.0),
        "classifier_dropout": {
            "vgg": 0.35,
            "resnet": 0.25,
            "googlenet": 0.35,
        }[name],
        "pretrained": False,
    }


def logits_from_output(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        output = output.logits
    elif isinstance(output, (tuple, list)):
        output = output[0]
    return output.reshape(-1)


def evaluate_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.inference_mode():
        for images, _ in loader:
            if device.type == "cuda":
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = logits_from_output(model(images))
            pieces.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.float32)


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def choose_validation_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [
        metrics_at_threshold(y_true, probabilities, float(t))["balanced_accuracy"]
        for t in candidates
    ]
    best_index = int(np.argmax(scored))
    return float(candidates[best_index]), float(scored[best_index])


def append_epoch_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def train_model(
    name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    csv_path: Path,
    split_hash: str,
    output_dir: Path,
    args,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / MODEL_FILES[name]
    last_path = model_dir / "last_training_state.pt"
    done_path = model_dir / "completed_summary.json"
    epoch_csv = model_dir / "epoch_metrics.csv"

    if done_path.is_file() and not args.force:
        previous = json.loads(done_path.read_text(encoding="utf-8"))
        logger.info(
            "[%s] Already completed; skipping. Existing result: %s",
            name, previous.get("validation_ba_at_05"),
        )
        return previous

    seed = SEED + {"vgg": 11, "resnet": 22, "googlenet": 33}[name]
    seed_everything(seed)

    train_dataset = UltrasoundDataset(train_df, csv_path, training=True)
    valid_dataset = UltrasoundDataset(valid_df, csv_path, training=False)

    workers = max(0, int(args.num_workers))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    config = model_config(
        name,
        args.dropblock,
        args.stochastic_depth,
    )
    model = build_model(
        name,
        args.dropblock,
        args.stochastic_depth,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    n_positive = int(train_df["label"].sum())
    n_negative = len(train_df) - n_positive
    pos_weight_value = (
        n_negative / max(1, n_positive)
    ) * float(args.pos_weight_multiplier)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    y_valid = valid_df["label"].to_numpy(dtype=int)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    wait = 0
    start_epoch = 1

    if last_path.is_file() and args.resume and not args.force:
        logger.info("[%s] Resuming from %s", name, last_path)
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get("split_sha256") != split_hash:
            raise ValueError(
                f"[{name}] Resume checkpoint split hash does not match current split."
            )
        if saved.get("model_config") != config:
            raise ValueError(
                f"[{name}] Resume checkpoint model config differs from this run."
            )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        best_state = saved.get("best_state_dict")
        best_score = float(saved.get("best_score", -1.0))
        best_epoch = int(saved.get("best_epoch", 0))
        wait = int(saved.get("wait", 0))
        start_epoch = int(saved.get("epoch", 0)) + 1

    logger.info("=" * 70)
    logger.info("[%s] Starting training", name)
    logger.info("[%s] Architecture: %s", name, config["architecture"])
    logger.info("[%s] Device: %s | AMP: %s", name, device, use_amp)
    if device.type == "cuda":
        logger.info("[%s] GPU: %s", name, torch.cuda.get_device_name(0))
    logger.info(
        "[%s] Train: %d images / %d patients | Valid: %d images / %d patients",
        name,
        len(train_df),
        train_df["patient_group"].nunique(),
        len(valid_df),
        valid_df["patient_group"].nunique(),
    )
    logger.info("[%s] pos_weight: %.4f", name, pos_weight_value)
    logger.info("[%s] Split SHA256: %s", name, split_hash)
    logger.info("[%s] Model config: %s", name, config)
    logger.info(
        "[%s] Training settings: epochs=%d, batch=%d, lr=%g, wd=%g, patience=%d",
        name,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.patience,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            if device.type == "cuda":
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = logits_from_output(model(images))
                loss = loss_fn(logits.float(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item()) * len(labels)
            total_seen += len(labels)

        probabilities = evaluate_probs(model, valid_loader, device, use_amp)
        ba05 = float(
            balanced_accuracy_score(
                y_valid,
                (probabilities >= 0.5).astype(int),
            )
        )
        auc = safe_auc(y_valid, probabilities)
        scheduler.step(ba05)

        improved = ba05 > best_score + 1e-6
        if improved:
            best_score = ba05
            best_epoch = epoch
            wait = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_payload = {
                "model_state_dict": best_state,
                "model_config": config,
                "image_size": list(IMAGE_SIZE),
                "mean": list(MEAN),
                "std": list(STD),
                "split_sha256": split_hash,
                "best_epoch": int(best_epoch),
                "validation_ba_at_05": float(best_score),
                "threshold": 0.5,
                "threshold_selected_on_validation": False,
                "training_config": {
                    "epochs_max": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "pos_weight_multiplier": float(args.pos_weight_multiplier),
                    "dropblock_prob": float(args.dropblock),
                    "stochastic_depth_prob": float(
                        args.stochastic_depth if name == "resnet" else 0.0
                    ),
                    "pretrained": False,
                    "augmentation": [
                        "RandomHorizontalFlip(p=0.5)",
                        "RandomRotation(degrees=5)",
                    ],
                },
            }
            atomic_torch_save(best_payload, best_path)
        else:
            wait += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        mean_loss = total_loss / max(1, total_seen)
        logger.info(
            "[%s] Epoch %02d/%02d | loss=%.5f | BA@0.5=%.5f | AUC=%.5f "
            "| LR=%.7g | best=%.5f@%d | wait=%d/%d",
            name,
            epoch,
            args.epochs,
            mean_loss,
            ba05,
            auc,
            current_lr,
            best_score,
            best_epoch,
            wait,
            args.patience,
        )

        append_epoch_csv(
            epoch_csv,
            {
                "model": name,
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_ba_at_05": ba05,
                "validation_auc": auc,
                "learning_rate": current_lr,
                "best_ba_at_05": best_score,
                "best_epoch": best_epoch,
                "wait": wait,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )

        # Save full resumable state after every epoch.
        last_payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_state_dict": best_state,
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "wait": int(wait),
            "model_config": config,
            "split_sha256": split_hash,
            "seed": int(seed),
        }
        atomic_torch_save(last_payload, last_path)

        if wait >= args.patience:
            logger.info(
                "[%s] Early stopping at epoch %d; best epoch %d.",
                name, epoch, best_epoch,
            )
            break

    if best_state is None:
        raise RuntimeError(f"[{name}] No valid best checkpoint was created.")

    # Restore the best epoch and evaluate once more.
    model.load_state_dict(best_state, strict=True)
    final_probs = evaluate_probs(model, valid_loader, device, use_amp)
    threshold, tuned_ba = choose_validation_threshold(y_valid, final_probs)
    fixed_metrics = metrics_at_threshold(y_valid, final_probs, 0.5)
    tuned_metrics = metrics_at_threshold(y_valid, final_probs, threshold)
    final_auc = safe_auc(y_valid, final_probs)

    # Store both fixed-threshold and validation-selected-threshold metrics.
    final_checkpoint = {
        "model_state_dict": best_state,
        "model_config": config,
        "image_size": list(IMAGE_SIZE),
        "mean": list(MEAN),
        "std": list(STD),
        "split_sha256": split_hash,
        "best_epoch": int(best_epoch),
        "validation_auc": float(final_auc),
        "validation_ba_at_05": float(fixed_metrics["balanced_accuracy"]),
        "validation_threshold": float(threshold),
        "validation_ba_at_selected_threshold": float(tuned_ba),
        "threshold": float(threshold),
        "threshold_selected_on_validation": True,
        "training_config": {
            "epochs_max": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight_multiplier": float(args.pos_weight_multiplier),
            "dropblock_prob": float(args.dropblock),
            "stochastic_depth_prob": float(
                args.stochastic_depth if name == "resnet" else 0.0
            ),
            "pretrained": False,
            "augmentation": [
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(degrees=5)",
            ],
        },
        "validation_metrics_at_05": fixed_metrics,
        "validation_metrics_at_selected_threshold": tuned_metrics,
    }
    atomic_torch_save(final_checkpoint, best_path)

    prediction_path = model_dir / "validation_predictions.csv"
    predictions = valid_df[
        ["sample_id", "patient_group", "label"]
    ].copy()
    predictions["probability_present"] = final_probs
    predictions["prediction_at_05"] = (final_probs >= 0.5).astype(int)
    predictions["prediction_at_selected_threshold"] = (
        final_probs >= threshold
    ).astype(int)
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    summary = {
        "model": name,
        "architecture": config["architecture"],
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "validation_auc": float(final_auc),
        "validation_ba_at_05": float(fixed_metrics["balanced_accuracy"]),
        "validation_threshold": float(threshold),
        "validation_ba_at_selected_threshold": float(tuned_ba),
        "validation_metrics_at_05": fixed_metrics,
        "validation_metrics_at_selected_threshold": tuned_metrics,
        "split_sha256": split_hash,
        "train_images": int(len(train_df)),
        "valid_images": int(len(valid_df)),
        "train_patients": int(train_df["patient_group"].nunique()),
        "valid_patients": int(valid_df["patient_group"].nunique()),
        "prediction_file": str(prediction_path),
        "note": (
            "Validation metrics only; not private-test performance. "
            "Threshold is selected once on the official validation set and may "
            "be optimistic. No ensemble or prediction averaging is used."
        ),
    }
    done_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Reload smoke test: reconstruct exactly the same single architecture.
    loaded = torch.load(best_path, map_location="cpu", weights_only=True)
    check_model = build_model(
        name,
        args.dropblock,
        args.stochastic_depth,
    )
    check_model.load_state_dict(loaded["model_state_dict"], strict=True)
    check_model.eval()
    sample_images, _ = next(iter(valid_loader))
    sample_images = sample_images[:2].to(device)
    model.eval()
    with torch.inference_mode():
        reference = logits_from_output(model(sample_images)).detach().cpu()
        reloaded_device_model = check_model.to(device).eval()
        reloaded_output = logits_from_output(
            reloaded_device_model(sample_images)
        ).detach().cpu()
    reload_max_diff = float((reference - reloaded_output).abs().max().item())
    if reload_max_diff > 1e-4:
        raise RuntimeError(
            f"[{name}] Checkpoint reload output mismatch: {reload_max_diff}"
        )

    logger.info("[%s] BEST EPOCH: %d", name, best_epoch)
    logger.info("[%s] Validation AUC: %.5f", name, final_auc)
    logger.info(
        "[%s] Validation BA@0.5: %.5f | threshold=%.3f | BA@threshold=%.5f",
        name,
        fixed_metrics["balanced_accuracy"],
        threshold,
        tuned_ba,
    )
    logger.info(
        "[%s] Confusion matrix @0.5 [[TN,FP],[FN,TP]]: [[%d,%d],[%d,%d]]",
        name,
        fixed_metrics["tn"],
        fixed_metrics["fp"],
        fixed_metrics["fn"],
        fixed_metrics["tp"],
    )
    logger.info(
        "[%s] Reload test: OK | max logit difference=%.8g",
        name,
        reload_max_diff,
    )
    logger.info("[%s] Saved checkpoint: %s", name, best_path)
    logger.info("[%s] Saved predictions: %s", name, prediction_path)
    return summary


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return MODEL_ORDER
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    invalid = sorted(set(names) - set(MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Choose from {MODEL_ORDER}.")
    # Keep canonical order and avoid duplicate runs.
    return [name for name in MODEL_ORDER if name in set(names)]


def main():
    parser = argparse.ArgumentParser(
        description="Train Task 2 VGG11, ResNet18, and GoogLeNet independently."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default="all",
                        help="'all' or comma-separated: vgg,resnet,googlenet")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--pos-weight-multiplier", type=float, default=0.80)
    parser.add_argument("--dropblock", type=float, default=0.08)
    parser.add_argument(
        "--stochastic-depth",
        type=float,
        default=0.0,
        help="Default 0.0 for conservative ensemble-rule compliance; "
             "try 0.10 only if instructor confirms Stochastic Depth is allowed.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--force", action="store_true",
                        help="Rerun completed models and ignore existing resume state.")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    if not 0.0 <= args.dropblock < 1.0:
        raise ValueError("dropblock must be in [0,1).")
    if not 0.0 <= args.stochastic_depth < 1.0:
        raise ValueError("stochastic-depth must be in [0,1).")

    csv_path = Path(args.csv).expanduser().resolve()
    split_path = Path(args.split_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

    df = read_labels(csv_path)
    train_idx, valid_idx, split_hash = load_shared_split(df, split_path)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    logger.info("=== TASK 2 STRUCTURED-REGULARIZATION TRAINING ===")
    logger.info("CSV: %s", csv_path)
    logger.info("Split: %s", split_path)
    logger.info("Split SHA256: %s", split_hash)
    logger.info(
        "Dataset: %d images | %d patients | labels=%s",
        len(df),
        df["patient_group"].nunique(),
        df["label"].value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Train: %d images/%d patients | Valid: %d images/%d patients",
        len(train_df),
        train_df["patient_group"].nunique(),
        len(valid_df),
        valid_df["patient_group"].nunique(),
    )
    logger.info(
        "Validation labels: %s",
        valid_df["label"].value_counts().sort_index().to_dict(),
    )
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Models to train: %s", parse_models(args.models))
    logger.info(
        "Regularization: DropBlock=%.3f; Stochastic Depth=%.3f",
        args.dropblock,
        args.stochastic_depth,
    )
    logger.info(
        "No pretrained weights, no external data, no ensemble, no prediction averaging."
    )

    # Preflight every image path before spending time on model training.
    logger.info("Checking image paths...")
    resolved_paths = [
        resolve_image_path(value, csv_path)
        for value in df["path"].astype(str)
    ]
    missing = [str(path) for path in resolved_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing. First examples: {missing[:5]}"
        )
    logger.info("Image path check passed: %d/%d", len(resolved_paths), len(df))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name in parse_models(args.models):
        try:
            result = train_model(
                name=name,
                train_df=train_df,
                valid_df=valid_df,
                csv_path=csv_path,
                split_hash=split_hash,
                output_dir=output_dir,
                args=args,
                device=device,
                logger=logger,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("[%s] FAILED: %s", name, exc)
            failures.append({
                "model": name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            # Continue to the next required architecture so an isolated
            # model-specific issue does not waste the whole unattended run.
            continue

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_sha256": split_hash,
        "device": str(device),
        "models_requested": parse_models(args.models),
        "completed": results,
        "failures": failures,
        "note": (
            "All three architectures are independent models. Their predictions "
            "must not be averaged for this assignment. Validation scores are "
            "not private-test scores."
        ),
    }
    summary_path = output_dir / "task2_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info("TASK 2 RUN FINISHED")
    for item in results:
        logger.info(
            "%s | BA@0.5=%.5f | tuned threshold=%.3f | tuned BA=%.5f | %s",
            item["model"],
            item["validation_ba_at_05"],
            item["validation_threshold"],
            item["validation_ba_at_selected_threshold"],
            item["checkpoint"],
        )
    for item in failures:
        logger.error("%s failed: %s", item["model"], item["error"])
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
