import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_loaders
from train_utils import fit


def build_model(pretrained=True):
    if pretrained:
        m = models.googlenet(weights="IMAGENET1K_V1")
        m.aux_logits = False
        m.aux1 = None
        m.aux2 = None
    else:
        m = models.googlenet(weights=None, aux_logits=False, init_weights=True, transform_input=True)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/T2_data"
    out = sys.argv[2] if len(sys.argv) > 2 else "googlenet.pt"
    pretrained = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"
    torch.manual_seed(42)
    tl, vl, y = get_loaders(root, str(Path(__file__).resolve().parent.parent / "split.csv"))
    fit(build_model(pretrained), tl, vl, y, out, epochs=15, lr=1e-4 if pretrained else 3e-4)
