# linux/arm64 native. An x86-only base would run under QEMU at roughly a
# fifth of native speed, which shows up as slowness rather than an error.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision==0.23.0

# 170 MB, against a base image already over 1 GB.
RUN python -c "from torchvision import datasets; \
    datasets.CIFAR10('/data', train=True, download=True); \
    datasets.CIFAR10('/data', train=False, download=True)"

COPY train.py /app/train.py

ENV DATA_DIR=/data \
    CHECKPOINT_DIR=/checkpoints \
    PYTHONUNBUFFERED=1

CMD ["python", "/app/train.py"]
