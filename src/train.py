"""MNIST CNN with checkpoint/resume, sized for a CPU-only Kubernetes pod.

Every epoch writes a checkpoint. On start, the script looks for the newest
checkpoint and resumes from it. Delete the pod mid-run and the replacement
picks up where it stopped, which is the behaviour the Job manifest is there
to demonstrate.
"""

import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = int(os.environ.get("SEED", "0"))
THREADS = int(os.environ.get("TORCH_THREADS", "0"))

CKPT_PATH = CKPT_DIR / "latest.pt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Net(nn.Module):
    """Two conv blocks into a classifier. About 1.2M parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def loaders():
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(DATA_DIR, train=True, download=False, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=False, transform=tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=1000, num_workers=2),
    )


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def save(model, opt, epoch, acc):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "accuracy": acc,
        },
        tmp,
    )
    tmp.replace(CKPT_PATH)  # atomic, so a kill mid-write cannot corrupt it


def main():
    torch.manual_seed(SEED)
    if THREADS:
        torch.set_num_threads(THREADS)
    log(f"torch {torch.__version__}, {torch.get_num_threads()} threads")

    # SIGTERM is what kubectl delete sends. Exit cleanly so the checkpoint
    # written at the end of the last epoch stays the newest valid one.
    signal.signal(signal.SIGTERM, lambda *_: (log("SIGTERM, exiting"), sys.exit(0)))

    model, start = Net(), 0
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["epoch"]
        log(f"resumed from epoch {start} (accuracy {ckpt['accuracy']:.2f}%)")
    else:
        log("no checkpoint, starting fresh")

    if start >= EPOCHS:
        log(f"already trained {start}/{EPOCHS} epochs, nothing to do")
        return

    train_loader, test_loader = loaders()

    for epoch in range(start + 1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            if i % 100 == 0:
                log(f"epoch {epoch}  batch {i}/{len(train_loader)}  loss {loss.item():.4f}")
        acc = evaluate(model, test_loader)
        save(model, opt, epoch, acc)
        log(
            f"epoch {epoch}/{EPOCHS} done in {time.time() - t0:.1f}s  "
            f"loss {total / len(train_loader):.4f}  test accuracy {acc:.2f}%  checkpointed"
        )

    log("training complete")


if __name__ == "__main__":
    main()
