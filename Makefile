# Cluster lifecycle and the MNIST training loop.
# `make help` lists targets.

PROFILE  ?= nn-train
IMAGE    ?= my-nn:dev
EPOCHS   ?= 5

.PHONY: help up down destroy status build load deploy logs watch clean reset train

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Create or resume the cluster (prints the full start command)
	@if minikube status -p $(PROFILE) >/dev/null 2>&1; then \
		echo "resuming existing cluster"; minikube start -p $(PROFILE); \
	else \
		echo "creating cluster:"; \
		echo "  minikube start -p $(PROFILE) --cpus=12 --memory=18g --disk-size=80g \\"; \
		echo "    --extra-config=kubelet.system-reserved=memory=6Gi,cpu=2"; \
		minikube start -p $(PROFILE) --cpus=12 --memory=18g --disk-size=80g \
			--extra-config=kubelet.system-reserved=memory=6Gi,cpu=2; \
	fi
	@minikube profile $(PROFILE)

down: ## Stop the cluster, keep its state and checkpoints
	minikube stop -p $(PROFILE)

destroy: ## Delete the cluster and everything in it
	minikube delete -p $(PROFILE)

status: ## Show node capacity against the container's real cgroup ceiling
	@kubectl get nodes
	@echo
	@echo "allocatable: $$(kubectl get node $(PROFILE) -o jsonpath='{.status.allocatable.cpu}') cpu, \
$$(kubectl get node $(PROFILE) -o jsonpath='{.status.allocatable.memory}' | sed 's/Ki//' | awk '{printf "%.2f GiB", $$1/1048576}')"
	@echo "cgroup cap:  $$(docker inspect $(PROFILE) --format '{{.HostConfig.NanoCpus}}' | awk '{print $$1/1000000000}') cpu, \
$$(docker inspect $(PROFILE) --format '{{.HostConfig.Memory}}' | awk '{printf "%.2f GiB", $$1/1073741824}')"
	@echo "(allocatable must stay below the cgroup cap)"

build: ## Build the training image on the host
	docker build -t $(IMAGE) .

load: build ## Push the image into the cluster's containerd store
	minikube image load $(IMAGE) -p $(PROFILE)

deploy: ## Create the PVC and start the training Job
	kubectl apply -f k8s/pvc.yaml
	sed 's/value: "5"/value: "$(EPOCHS)"/' k8s/job.yaml | kubectl apply -f -

train: load deploy watch ## Build, load, deploy and follow in one step

logs: ## Tail the training pod
	kubectl logs -l job-name=mnist-train -f --tail=50

watch: ## Block until the Job finishes, then print the epoch summaries
	@until kubectl get job mnist-train -o jsonpath='{.status.succeeded}' 2>/dev/null | grep -q 1; do sleep 5; done
	@kubectl logs -l job-name=mnist-train --tail=200 | grep -vE 'batch [1-9]'

clean: ## Delete the Job, keep checkpoints on the PVC
	-kubectl delete job mnist-train

reset: clean ## Delete the Job and wipe checkpoints, forcing a fresh run
	-kubectl delete pvc mnist-checkpoints
