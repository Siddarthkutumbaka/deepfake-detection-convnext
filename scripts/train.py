import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.dfdc.utils import set_seed, pick_device, ensure_dir
from src.dfdc.modeling import build_model

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_dir", default=None, help="Optional: path to train ImageFolder")
    ap.add_argument("--val_dir", default=None, help="Optional: path to val ImageFolder")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    print("Device:", device)

    model, transform = build_model(cfg["train"]["model_name"], num_classes=2)
    model.to(device)

    # Expect ImageFolder style: train_dir/{REAL,FAKE}/... and val_dir/{REAL,FAKE}/...
    train_dir = args.train_dir
    val_dir = args.val_dir
    if train_dir is None or val_dir is None:
        raise SystemExit(
            "Pass --train_dir and --val_dir (ImageFolder folders). "
            "Example: --train_dir data/train_frames --val_dir data/val_frames"
        )

    train_ds = ImageFolder(train_dir, transform=transform)
    val_ds = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    ensure_dir(cfg["train"]["save_dir"])
    best_acc = -1.0
    best_path = os.path.join(cfg["train"]["save_dir"], cfg["train"]["best_name"])

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best: {best_path} (val_acc={best_acc:.4f})")

    print("Done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()
