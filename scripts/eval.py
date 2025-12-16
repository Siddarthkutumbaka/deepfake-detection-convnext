import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from src.dfdc.utils import pick_device
from src.dfdc.modeling import build_model

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        p_fake = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p_fake)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best_model.pth")
    ap.add_argument("--val_dir", required=True, help="ImageFolder val folder")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = pick_device(cfg.get("device", "auto"))
    print("Device:", device)

    model, transform = build_model(cfg["train"]["model_name"], num_classes=2)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    ds = ImageFolder(args.val_dir, transform=transform)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    probs, labels = predict_probs(model, loader, device)
    thresh = cfg["eval"].get("threshold", 0.5)
    preds = (probs >= thresh).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    cm = confusion_matrix(labels, preds)

    print("ROC-AUC:", auc)
    print("Threshold:", thresh)
    print("Confusion Matrix:\n", cm)
    print("\nReport:\n", classification_report(labels, preds, target_names=["REAL", "FAKE"]))

if __name__ == "__main__":
    main()
