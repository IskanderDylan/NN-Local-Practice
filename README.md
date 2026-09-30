# local-nn

A reproducible single-node Kubernetes cluster on macOS for training neural networks locally, built on minikube over Docker Desktop.

Start from a clean machine and follow this top to bottom. If minikube and Docker are already installed, skip to [Configure Docker Desktop](#2-configure-docker-desktop).

## Verified on

| Component | Version |
|---|---|
| macOS | Darwin 26.6.2 (arm64) |
| Hardware | Apple M5 Max, 18 cores, 36 GB RAM |
| Homebrew | 7.0.7 |
| Docker Desktop | 29.8.1 |
| minikube | v1.39.0 |
| kubectl | v1.37.1 |
| Kubernetes | v1.37.0 |
| Container runtime | containerd 2.3.4 |

The sizing numbers below assume 36 GB of RAM. See [Sizing for your machine](#sizing-for-your-machine) to scale them.

## 1. Install the toolchain

### Homebrew

Skip if `brew --version` already answers.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, Homebrew installs to `/opt/homebrew` and the installer prints two `eval` lines to add it to your PATH. Run them, then reopen your shell.

### Docker Desktop

```bash
brew install --cask docker
```

Launch Docker Desktop from Applications once after installing. It has to complete its first-run setup and start its Linux VM before minikube can use it. Wait for the whale icon in the menu bar to stop animating, then confirm:

```bash
docker version
```

### minikube and kubectl

```bash
brew install minikube kubectl
```

## 2. Configure Docker Desktop

minikube runs as a container inside Docker Desktop's Linux VM, so the VM's memory ceiling caps the cluster. Docker Desktop ships with roughly 8 GB allocated, which will not hold a training workload.

Open Docker Desktop, go to **Settings > Resources**, and set:

| Setting | Value | Why |
|---|---|---|
| Memory | 20 GB | Leaves 16 GB for macOS |
| CPUs | 14 | Leaves 4 cores for the host |
| Swap | 2 GB | Default is fine |
| Disk image size | 100 GB | ML images run 2 to 6 GB each and accumulate |

Click **Apply & Restart** and wait for Docker to come back. Verify the VM picked up the change:

```bash
docker info --format 'CPUs: {{.NCPU}}  Mem: {{.MemTotal}}'
```

Memory reports in bytes. Anything near 21474836480 means the 20 GB took effect.

Skipping this step produces `Requested memory allocation 18432MB is more than your system limit` when you start the cluster.

## 3. Configure minikube defaults

These write to `~/.minikube/config/config.json` and apply to every cluster you create afterward.

```bash
minikube config set driver docker
minikube config set container-runtime containerd
minikube config set cpus 10
minikube config set memory 14g
minikube config set disk-size 60g
```

Review them with `minikube config view`. Remove one with `minikube config unset <key>`.

Two caveats worth knowing before you rely on these:

- They apply at cluster creation. Changing a value and restarting an existing cluster does nothing, because minikube bakes the original values into `~/.minikube/profiles/<name>/config.json`. Resizing requires `minikube delete` followed by `minikube start`.
- Command-line flags override the config file, which is what the start command in the next section does.

## 4. Start the cluster

```bash
minikube start -p nn-train \
  --cpus=12 \
  --memory=18g \
  --disk-size=80g \
  --extra-config=kubelet.system-reserved=memory=6Gi,cpu=2

minikube profile nn-train
```

The second command points both minikube and kubectl at this cluster so you can drop `-p nn-train` from later commands.

Save this command. `--extra-config` lives in the profile rather than in `~/.minikube/config/config.json`, so `minikube config set` cannot carry it. Delete the cluster without re-passing the flag and you silently lose the protection described next.

### Why system-reserved is not optional

`/proc/meminfo` is not namespaced, so kubelet running inside the minikube container reads the Docker VM's totals instead of its own cgroup limit. The scheduler ends up believing it has more memory than the container can actually use.

Without the flag, on a 20 GB VM:

| | CPU | Memory |
|---|---|---|
| Requested at start | 12 | 18.00 GiB |
| Container cgroup limit | 12 | 18.00 GiB |
| Reported as allocatable | 14 | 23.44 GiB |

The scheduler hands out memory up to 23.44 GiB while the kernel kills the container at 18. That kill lands on the container, not the pod, so kubelet, etcd and the API server die together and the cluster drops mid-run.

With `system-reserved=memory=6Gi,cpu=2`:

| | CPU | Memory |
|---|---|---|
| Node capacity | 14 | 23.44 GiB |
| Allocatable | 12 | 17.43 GiB |
| Container cgroup limit | 12 | 18.00 GiB |

Allocatable now sits below the cgroup ceiling with room to spare. A pod that will not fit gets rejected at scheduling time.

## 5. Verify

```bash
kubectl get nodes
kubectl get pods -A
```

Expect `nn-train` in `Ready` state within 30 seconds, and eight pods in `kube-system` all `Running`: coredns, etcd, kindnet, kube-apiserver, kube-controller-manager, kube-proxy, kube-scheduler, storage-provisioner.

Confirm allocatable came out under the cgroup limit:

```bash
kubectl get node nn-train -o jsonpath='{.status.allocatable.memory}{"\n"}'
docker inspect nn-train --format '{{.HostConfig.Memory}}'
```

The first prints KiB, the second bytes. The first value has to be smaller. At the sizes above you get 18277916Ki (17.43 GiB) against 19327352832 bytes (18.00 GiB).

## 6. Build and load images

This cluster runs containerd, so `eval $(minikube docker-env)` does not apply. No dockerd runs inside the node. Build with your local Docker and push the result into the cluster:

```bash
docker build -t my-nn:dev .
minikube image load my-nn:dev
```

Set `imagePullPolicy: Never` in the pod spec so Kubernetes uses the loaded image instead of reaching for a registry:

```yaml
spec:
  containers:
    - name: train
      image: my-nn:dev
      imagePullPolicy: Never
```

List what the cluster holds with `minikube image ls`.

containerd costs you a slower image loop than the docker runtime would, and buys you a runtime that matches what production clusters run, so manifests that work here keep working when you move off your laptop.

## 7. Teardown and recreate

```bash
minikube stop -p nn-train     # keep the cluster, free the memory
minikube start -p nn-train    # resume it
minikube delete -p nn-train   # destroy it
```

`stop` and `start` preserve cluster state. `delete` discards everything, including anything written to a PVC, so keep manifests in git and training data outside the cluster.

Recreating means running the full command from [Start the cluster](#4-start-the-cluster) again, `--extra-config` included.

## Sizing for your machine

Give macOS at least 14 to 16 GB and leave 4 cores outside Docker. From there:

| Host RAM | Docker Desktop | `--memory` | `--cpus` | `system-reserved` |
|---|---|---|---|---|
| 16 GB | 10 GB | 8g | 4 | memory=3Gi,cpu=1 |
| 24 GB | 14 GB | 12g | 6 | memory=4Gi,cpu=1 |
| 36 GB | 20 GB | 18g | 12 | memory=6Gi,cpu=2 |
| 64 GB | 40 GB | 36g | 16 | memory=10Gi,cpu=2 |

Two rules generate the whole table. Keep `--memory` a few GB under the Docker Desktop allocation so the VM has room for its own overhead. Size `system-reserved` memory so that (Docker Desktop total) minus (reserved) lands under `--memory`, which is what keeps allocatable below the cgroup ceiling.

## Known limits

**No GPU.** Apple Silicon does not pass Metal through to Linux containers, under minikube or Docker Desktop or anything else. Training here runs on CPU. That works for MNIST-scale models and for proving out your Kubernetes plumbing. For a real training run, use MPS natively on macOS or rent a remote GPU, and treat this cluster as the orchestration rehearsal.

**Build for arm64.** Images have to target `linux/arm64` or they run under QEMU emulation at roughly a fifth of native speed. `pytorch/pytorch` and `python:3.12-slim` both publish arm64 variants. An x86-only base image gives you terrible performance rather than an error, which makes it hard to spot.

**Single node.** One node carries the control plane and your workloads together. A job that exhausts memory degrades the control plane with it, which is the failure `system-reserved` exists to prevent.
