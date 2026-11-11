"""Fine-tune ResNet18 on the Fruit Recognition dataset."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import models

from common import make_loaders


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    preds, labels, loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += criterion(out, y).item() * y.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (preds == labels).mean(), f1_score(labels, preds, average="macro"), loss / len(labels), preds, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train/train")
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes, train_samples = make_loaders(args.data, args.batch_size)
    print(f"{len(classes)} classes | train={len(train_samples)} | val={len(val_loader.dataset)} | {device}")
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(classes, f)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_path = -1.0, os.path.join(args.out, "best_model.pth")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        acc, f1, vloss, preds, labels = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{args.epochs} train_loss={running/len(train_loader):.4f} "
              f"val_loss={vloss:.4f} val_acc={acc:.4f} val_macroF1={f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            np.savetxt(os.path.join(args.out, "confusion_matrix.csv"),
                       confusion_matrix(labels, preds), fmt="%d", delimiter=",")

    print(f"Best val accuracy: {best_acc:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
