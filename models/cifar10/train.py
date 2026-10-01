"""CIFAR-10 ResNet-9 with checkpoint/resume, sized for a CPU-only pod.

Same checkpoint contract as the MNIST workload: one checkpoint per epoch,
resume from the newest on start. This one runs long enough that a mid-run
pod deletion is worth trying.
"""

import os
import signal
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LR = float(os.environ.get("LR", "2e-3"))
SEED = int(os.environ.get("SEED", "0"))
THREADS = int(os.environ.get("TORCH_THREADS", "0"))

CKPT_PATH = CKPT_DIR / "latest.pt"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def conv_block(cin, cout, pool=False):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """The DAWNBench speedrun architecture. About 6.6M parameters."""

    def __init__(self, classes=10):
        super().__init__()
        self.prep = conv_block(3, 64)
        self.layer1 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.layer2 = conv_block(128, 256, pool=True)
        self.layer3 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x) + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x) + x
        return self.head(x)


def loaders():
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=train_tf)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=test_tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=512, num_workers=2),
    )


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def save(model, opt, sched, epoch, acc):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epochs_planned": EPOCHS,
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

    signal.signal(signal.SIGTERM, lambda *_: (log("SIGTERM, exiting"), sys.exit(0)))

    train_loader, test_loader = loaders()

    model, start = ResNet9(), 0
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["epoch"]
        # OneCycleLR bakes total_steps into its state. Restoring a schedule
        # built for a different epoch budget makes it step past the end and
        # raise, so rebuild and fast-forward instead.
        if ckpt.get("epochs_planned") == EPOCHS:
            sched.load_state_dict(ckpt["scheduler"])
        else:
            log(
                f"epoch budget changed ({ckpt.get('epochs_planned')} -> {EPOCHS}), "
                "rebuilding the LR schedule"
            )
            # Fast-forwarding steps the schedule without matching optimizer
            # steps, which torch warns about. Expected here, so keep it out
            # of the log.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for _ in range(start * len(train_loader)):
                    sched.step()
        log(f"resumed from epoch {start} (accuracy {ckpt['accuracy']:.2f}%)")
    else:
        log("no checkpoint, starting fresh")

    if start >= EPOCHS:
        log(f"already trained {start}/{EPOCHS} epochs, nothing to do")
        return

    for epoch in range(start + 1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if i % 50 == 0:
                log(f"epoch {epoch}  batch {i}/{len(train_loader)}  loss {loss.item():.4f}")
        acc = evaluate(model, test_loader)
        save(model, opt, sched, epoch, acc)
        log(
            f"epoch {epoch}/{EPOCHS} done in {time.time() - t0:.1f}s  "
            f"loss {total / len(train_loader):.4f}  test accuracy {acc:.2f}%  checkpointed"
        )

    log("training complete")


if __name__ == "__main__":
    main()
