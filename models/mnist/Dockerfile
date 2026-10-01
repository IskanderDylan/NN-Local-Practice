# linux/arm64 native. An x86-only base would run under QEMU at roughly a
# fifth of native speed, which shows up as slowness rather than an error.
FROM python:3.12-slim

WORKDIR /app

# On aarch64 the default PyPI wheel is already the CPU build, so there is no
# CUDA payload to strip out. Pinned so a rebuild six months from now produces
# the same image.
RUN pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision==0.23.0

# Bake the dataset in at build time. The build runs on the host where the
# network works; the pod may have no egress at all. 11 MB compressed.
RUN python -c "from torchvision import datasets; \
    datasets.MNIST('/data', train=True, download=True); \
    datasets.MNIST('/data', train=False, download=True)"

COPY train.py /app/train.py

ENV DATA_DIR=/data \
    CHECKPOINT_DIR=/checkpoints \
    PYTHONUNBUFFERED=1

CMD ["python", "/app/train.py"]
