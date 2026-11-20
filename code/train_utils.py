import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score


def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            ps.append(torch.softmax(out, 1)[:, 1].cpu())
            ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def best_threshold(p, y):
    ts = np.linspace(0.05, 0.95, 91)
    s = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts]
    i = int(np.argmax(s))
    return float(ts[i]), float(s[i])


def fit(model, train_loader, valid_loader, train_labels, save_path, epochs=20, lr=3e-4, device="cuda"):
    model.to(device)
    counts = np.bincount(train_labels, minlength=2)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = -1
    for ep in range(epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        p, yv = predict(model, valid_loader, device)
        ba05 = balanced_accuracy_score(yv, (p >= 0.5).astype(int))
        t, ba = best_threshold(p, yv)
        print(f"ep {ep+1:02d} loss {total/len(train_loader.dataset):.4f} ba@0.5 {ba05:.4f} best_ba {ba:.4f} t {t:.2f}", flush=True)
        if ba > best:
            best = ba
            torch.save({"state_dict": model.state_dict(), "threshold": t, "valid_ba": ba, "epoch": ep + 1}, save_path)
    print("best", best)
    return best
