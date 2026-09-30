# local-nn

A reproducible single-node Kubernetes cluster on macOS for training neural networks locally, built on minikube over Docker Desktop. Ships with an MNIST CNN that trains to 99% in 83 seconds, set up to demonstrate checkpointing, resume across pod deletion, and resource limits the scheduler can enforce.

Start from a clean machine and follow this top to bottom. If Docker and minikube are already installed and sized, the whole thing is:

```bash
make up      # create the cluster
make train   # build, load, run MNIST, follow the logs
```

If minikube and Docker are already installed, skip to [Configure Docker Desktop](#2-configure-docker-desktop).

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

|  | CPU | Memory |
|---|---|---|
| Requested at start | 12 | 18.00 GiB |
| Container cgroup limit | 12 | 18.00 GiB |
| Reported as allocatable | 14 | 23.44 GiB |

The scheduler hands out memory up to 23.44 GiB while the kernel kills the container at 18. That kill lands on the container, not the pod, so kubelet, etcd and the API server die together and the cluster drops mid-run.

With `system-reserved=memory=6Gi,cpu=2`:

|  | CPU | Memory |
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

## 6. Run the MNIST workload

Once the cluster is `Ready`:

```bash
make train
```

That builds the image, loads it into the cluster, creates the PVC, starts the Job and follows it to completion. Expect this on the reference machine:

```
[21:56:46] torch 2.8.0+cpu, 4 threads
[21:56:46] no checkpoint, starting fresh
[21:56:59] epoch 1/5 done in 13.1s  loss 0.2046  test accuracy 98.28%  checkpointed
[21:57:11] epoch 2/5 done in 12.4s  loss 0.0650  test accuracy 98.61%  checkpointed
[21:57:24] epoch 3/5 done in 12.4s  loss 0.0445  test accuracy 99.03%  checkpointed
[21:57:36] epoch 4/5 done in 12.4s  loss 0.0358  test accuracy 99.09%  checkpointed
[21:57:48] epoch 5/5 done in 12.5s  loss 0.0307  test accuracy 99.06%  checkpointed
[21:57:48] training complete
```

Around 12.5 seconds per epoch on 4 CPU cores, 83 seconds for the Job. The model is a two-block CNN with roughly 1.2M parameters, which reaches 99% on MNIST and asks nothing of the machine.

Run more epochs with `make train EPOCHS=10`. Wipe the checkpoint and start over with `make reset`.

### Make targets

| Target | What it does |
|---|---|
| `make up` | Create the cluster, or resume it if it exists |
| `make down` | Stop the cluster, keeping checkpoints |
| `make destroy` | Delete the cluster and everything in it |
| `make status` | Print allocatable against the cgroup ceiling |
| `make build` | Build the image on the host |
| `make load` | Build, then push into the cluster's containerd store |
| `make deploy` | Create the PVC and start the Job |
| `make train` | load + deploy + watch |
| `make logs` | Follow the training pod |
| `make clean` | Delete the Job, keep checkpoints |
| `make reset` | Delete the Job and the PVC, forcing a fresh run |

`make up` prints the full `minikube start` command before running it, so the `--extra-config` flag stays visible rather than hiding behind the target.

### Layout

```
Dockerfile        python:3.12-slim, torch 2.8.0 CPU, dataset baked in
Makefile          cluster lifecycle and training targets
src/train.py      CNN, training loop, checkpoint and resume
k8s/pvc.yaml      2Gi claim that outlives the pod
k8s/job.yaml      the training Job, resource limits and thread count
```

## 7. How images reach the cluster

This cluster runs containerd, so `eval $(minikube docker-env)` does not apply. No dockerd runs inside the node. Build with your local Docker and push the result in:

```bash
docker build -t my-nn:dev .
minikube image load my-nn:dev
```

`k8s/job.yaml` sets `imagePullPolicy: Never` so Kubernetes uses the loaded image instead of reaching for Docker Hub. Drop that line and you get `ImagePullBackOff`, because `my-nn:dev` exists nowhere but your machine.

Check what the cluster holds with `minikube image ls`.

Loading a 1.19 GB image takes a few seconds, which is slower than the docker runtime's zero-copy `docker-env` trick. In exchange you get the runtime production clusters actually run, so manifests that work here keep working when you move off your laptop.

## 8. Where the dataset lives

The Dockerfile downloads MNIST at build time and bakes it into the image:

```dockerfile
RUN python -c "from torchvision import datasets; \
    datasets.MNIST('/data', train=True, download=True); \
    datasets.MNIST('/data', train=False, download=True)"
```

`src/train.py` then loads it with `download=False`, so a pod that somehow starts without the data fails loudly instead of quietly reaching for the network.

Three reasons this beats making each person fetch the files by hand:

- The build runs on your host, where the network works. A pod may have no egress at all, and debugging that teaches you nothing about Kubernetes.
- MNIST is 11 MB compressed. It adds nothing meaningful to a 1.19 GB image.
- A baked dataset makes the image self-contained, so `minikube image load` is the only transfer step and a run is reproducible from the image digest alone.

Manual download earns its place once the data outgrows the image. At that point the pattern changes: an `initContainer` fetches the dataset onto a shared PVC, the training container waits for it, and many Jobs reuse one copy. Worth building when you move to CIFAR-10 or larger. For MNIST it is friction without a lesson.

## 9. Exercises

The manifests are set up so these fail in instructive ways.

**Resume from a checkpoint.** Run five epochs, then ask for eight:

```bash
make train EPOCHS=5
make clean
make train EPOCHS=8
```

The second run prints `resumed from epoch 5 (accuracy 99.06%)` and trains only 6 through 8. The PVC survived the Job deletion.

**Kill a pod mid-epoch.** Start a long run, then delete the pod while it works:

```bash
make train EPOCHS=20 &
kubectl delete pod -l job-name=mnist-train
```

`backoffLimit: 3` gives the Job a new pod, which mounts the same claim and restarts from the last completed epoch. `train.py` writes checkpoints atomically through a temp file and rename, so a kill during a write cannot leave a corrupt file.

**Ask for more than the node has.** Raise the memory request past allocatable:

```bash
kubectl patch job mnist-train --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/memory","value":"20Gi"}]'
```

The pod stays `Pending` with `Insufficient memory`. Without `system-reserved` the scheduler would have admitted it against the VM's 23.44 GiB and let the kernel kill the node instead.

**Change the thread count.** `TORCH_THREADS` in `k8s/job.yaml` is set to 4 to match the CPU limit. Set it to 12 while leaving `limits.cpu` at 4 and the epoch time gets worse, because torch spawns 12 threads that fight over a quota of 4 cores.

## 10. Teardown and recreate

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
