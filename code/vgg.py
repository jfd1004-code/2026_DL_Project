import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_loaders
from train_utils import fit


def build_model(pretrained=True):
    m = models.vgg11_bn(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    return m


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/T2_data"
    out = sys.argv[2] if len(sys.argv) > 2 else "vgg.pt"
    torch.manual_seed(42)
    tl, vl, y = get_loaders(root, str(Path(__file__).resolve().parent.parent / "split.csv"))
    fit(build_model(), tl, vl, y, out, epochs=15, lr=1e-5)
