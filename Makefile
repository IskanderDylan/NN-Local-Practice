# Cluster lifecycle and training workloads.
# `make help` lists targets. `make models` lists what you can train.
#
# Pick a workload with MODEL=<name>, default mnist:
#   make train MODEL=cifar10 EPOCHS=20

PROFILE ?= nn-train
MODEL   ?= mnist
DIR      = models/$(MODEL)
IMAGE    = $(MODEL)-nn:dev
JOB      = $(MODEL)-train

.PHONY: help models up down destroy status build load deploy logs watch clean reset train

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  MODEL=$(MODEL) (override with MODEL=<name>, see 'make models')"

models: ## List available workloads
	@echo "  \033[36mmnist\033[0m      CNN, ~1.2M params, 99% in ~85s on 4 cores"
	@echo "  \033[36mcifar10\033[0m    ResNet-9, ~6.6M params, ~4.8 min/epoch on 8 cores"

check:
	@test -d $(DIR) || { echo "no such model '$(MODEL)'. try: make models"; exit 1; }

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
	@echo
	@kubectl get jobs,pvc 2>/dev/null

build: check ## Build the selected model's image
	docker build -t $(IMAGE) $(DIR)

load: build ## Push the image into the cluster's containerd store
	minikube image load $(IMAGE) -p $(PROFILE)

deploy: check ## Create the PVC and start the Job
	kubectl apply -f $(DIR)/pvc.yaml
ifdef EPOCHS
	@awk '/name: EPOCHS/{p=1} p&&/value:/{sub(/"[0-9]+"/,"\"$(EPOCHS)\"");p=0} {print}' \
		$(DIR)/job.yaml | kubectl apply -f -
else
	kubectl apply -f $(DIR)/job.yaml
endif

train: load deploy watch ## Build, load, deploy and follow in one step

logs: ## Tail the training pod
	kubectl logs -l job-name=$(JOB) -f --tail=50

watch: ## Block until the Job finishes, then print the epoch summaries
	@echo "waiting on $(JOB)..."
	@# caffeinate -i holds off idle sleep. Without it a long run stalls when
	@# the Mac sleeps: the pod keeps its state but stops getting CPU.
	@caffeinate -i sh -c 'until kubectl get job $(JOB) -o jsonpath="{.status.succeeded}" 2>/dev/null | grep -q 1; do sleep 5; done'
	@kubectl logs -l job-name=$(JOB) --tail=500 | grep -vE 'batch [0-9]+/'

clean: ## Delete the Job, keep checkpoints on the PVC
	-kubectl delete job $(JOB)

reset: clean ## Delete the Job and wipe this model's checkpoints
	-kubectl delete pvc $(MODEL)-checkpoints
