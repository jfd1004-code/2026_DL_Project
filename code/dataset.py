import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = (int(os.environ.get("IMG_H", 384)), int(os.environ.get("IMG_W", 256)))
AUG = os.environ.get("AUG", "base")


def get_transforms(train):
    if not train:
        return T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
    if AUG == "strong":
        return T.Compose([
            T.Resize(SIZE),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize(SIZE),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class SheathDataset(Dataset):
    def __init__(self, df, root, train):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r.path}").convert("RGB")
        return self.tf(img), torch.tensor(r.label, dtype=torch.long)


def get_loaders(root, split_csv, valid_fold=1, batch_size=32, num_workers=2):
    df = pd.read_csv(f"{root}/labels.csv").merge(pd.read_csv(split_csv)[["sample_id", "fold"]], on="sample_id")
    tr, va = df[df.fold != valid_fold], df[df.fold == valid_fold]
    if os.environ.get("FULL") == "1":
        tr = df
    train_loader = DataLoader(SheathDataset(tr, root, True), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(SheathDataset(va, root, False), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader, tr.label.values
