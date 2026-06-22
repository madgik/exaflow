# Exaflow Development deployment with Kubernetes in one node

## Configuration

The following packages need to be installed:

```
docker
kubectl
helm
```

kubectl [installation guide (Ubuntu)](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)<br />
helm [installation gude](https://helm.sh/docs/intro/install/)

## Data preparation

Data must be present within the defined data paths that each worker has.
Each worker on initialization load all preset data within the data path.

## Setup the kubernetes cluster with kind

1. Delete the existing cluster, if it exists

```
kind delete cluster
```

2. Create the cluster using the prod_env_tests setup (you can create a custom one if you want):

```
kind create cluster --config tests/prod_env_tests/deployment_configs/kind_configuration/kind_cluster.yaml
```

3. After the nodes are started, you need to taint them properly:
   <br /><br />The following 2 sets of commands taint and label the master (kubernetes) node. Both command sets, taint and label the same node (master kubernetes node), referenced with different name due to `kind` naming the nodes differently from time to time.

<br />Run command set 1:

```
kubectl taint nodes kind-control-plane node-role.kubernetes.io/master-
kubectl label node kind-control-plane nodeType=master
```

If the previous `kubectl` command gives an error about `kind-control-plane` not found, then run the command set 2:

```
kubectl taint nodes master node-role.kubernetes.io/master-
kubectl label node master nodeType=master
```

(you can also get the existing nodes with `kubectl get nodes`)<br />
<br />Taint and label the worker nodes

```
kubectl label node kind-worker nodeType=worker
kubectl label node kind-worker2 nodeType=worker
kubectl label node kind-worker3 nodeType=worker
```

4. (Optional) Build and load the docker images for the kubernetes cluster. If this step is omitted, the images will be pulled from dockerhub, which will most likely be slower, depending on the speed of your connection. Make sure the tag you build matches `exaflow_images.version` in `kubernetes/values.yaml` (or override it during `helm install`).

First, build the images:
<br />(you can execute these in separate terminals, concurrently)

```
docker build -f exaflow/worker/Dockerfile -t madgik/exaflow_worker:<TAG> ./
docker build -f exaflow/controller/Dockerfile -t madgik/exaflow_controller:<TAG> ./
docker build -f exaflow/aggregation_server/Dockerfile -t madgik/exaflow_aggregation_server:<TAG> ./
```

Second, load the docker images to the kuberentes cluster

```
kind load docker-image madgik/exaflow_worker:<TAG>
kind load docker-image madgik/exaflow_controller:<TAG>
kind load docker-image madgik/exaflow_aggregation_server:<TAG>
```

5. Deploy the Exaflow kubernetes pods using helm charts:

```
helm install exaflow kubernetes/ --set exaflow_images.version=<TAG>
```

6. (Validation) You can then run the prod_env_tests to see if the deployment is working (it might take about a minute for services to sync):

```
uv run pytest tests/prod_env_tests/
```
