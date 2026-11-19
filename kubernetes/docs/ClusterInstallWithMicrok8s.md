## Configuration with microk8s

`microk8s` to be installed on **master/worker** nodes:

```
sudo snap install microk8s --classic
```

Additional configuration needed on the **master** node only:

```
microk8s enable dns ingress helm3
```

(Optional) You can enable the k8s dashboard with:

```
microk8s enable dashboard
```

### (Optional) Cluster Configuration with Multiple Network Interfaces (e.g. VPN)

If the cluster is going to be configured on top of a **VPN**, or the k8s **master** node has more than one network interface, we need to select the proper network interface on top of which the cluster will communicate.

If this is not done properly, **it could result on worker node failing to join the cluster correctly**, leading to nodes appearing as `Not Ready`.

In order to configure microk8s on the **master** node to use the proper network interface, followe these steps:

1. Disable ingress with the command:

```
microk8s disable ingress
```

2. Append the following lines to the kubelet configuration files:

```
sudo echo --node-ip=<MASTER_WORKER_PROPER_INTERFACE_IP> >> /var/snap/microk8s/current/args/kubelet
sudo echo --advertise-address=<MASTER_WORKER_PROPER_INTERFACE_IP> >> /var/snap/microk8s/current/args/kube-apiserver
```

3. Restart MicroK8s with:

```
sudo snap restart microk8s
```

4. Finally, re-enable ingress with:

```
microk8s enable ingress
```

Following these steps, your Kubernetes **master** node should be properly configured.

For more details and reference, you can visit: [Kubernetes Bug Fix on GitHub](https://github.com/canonical/microk8s/issues/2402#issuecomment-1460214658)

## Cluster Management

### Configure the master node to run pods

Allow master-specific pods to run on the **master** node with:

```
microk8s kubectl taint nodes --all node-role.kubernetes.io/master-
microk8s kubectl label node <master-node-name> master=true
```

### Add a worker node to the cluster

1. On the **master** node, get the join token with the following command:

```
microk8s add-node
```

Use the provided command, including `--worker` on the **worker** node to join the cluster.

2. Allow worker-specific pods to run on the **worker** node with:

```
microk8s kubectl label node <worker-node-name> worker=true
```

3. If the node has status `Ready,SchedulingDisabled` run:

```
microk8s kubectl uncordon <node-name>
```

### Remove a worker node from the cluster

On the **master** node execute the following commands:

```
microk8s kubectl drain <node-name> --ignore-daemonsets
microk8s kubectl delete node <node-name>
```

On the **worker** node execute the following command:

```
microk8s leave
```

### Hostname configuration

Currently, each node is distinguished from another one using their hostnames and they <b>MUST</b> be alphanumeric.

If a node's hostname does not comply with that convention it has to be configured from the kubelet, for microk8s you can do the following, quoting from https://microk8s.io/docs/troubleshooting:

```
To fix this you can change the hostname or use the --hostname-override argument
in kubelet’s configuration in /var/snap/microk8s/current/args/kubelet.
```

**ATTENTION:** The hostname configuration should take place AFTER the node has joined the cluster.

## Images Registry Management

Due to the docker limit when deploying the cluster, the pods could fail with an `Imagepullbackoff` error that could be caused by:

```
You have reached your pull rate limit. You may increase the limit by authenticating and upgrading: <https://www.docker.com/increase-rate-limits.>
```

This can happen when many containers are pulled from the same ip reaching the docker pull limit, that will be reset the following day allowing you to pull images again.

A more permanent solution is to use a local registry. The k8s master node will keep a local image registry and the rest of the nodes will use that specific registry instead of docker's.

### In the master node follow these instructions:

Add the registry component in the microk8s cluster:

```
microk8s enable registry
```

Install docker so you can pull, tag and push images:

```
sudo snap install docker
```

Allow docker to connect to the private registry by adding in `/var/snap/docker/current/config/daemon.json` the following:

```
{
  "insecure-registries" : ["<master_node_ip>:32000"]
}
```

Restart docker to apply the change:

```
sudo snap restart docker
```

Login to docker to increase your limit:

```
docker login
```

Pull all the images needed:

```
sudo docker pull madgik/exaflow_controller:latest
sudo docker pull madgik/exaflow_worker:latest
sudo docker pull madgik/exaflow_aggregation_server:latest
sudo docker pull gpikra/coordinator:v7.0.7.4
sudo docker pull mongo:5.0.8
sudo docker pull redis:alpine3.15
```

Tag them using the private registry:

```
sudo docker tag madgik/exaflow_controller:latest <master_node_ip>:32000/exaflow_controller:latest
sudo docker tag madgik/exaflow_worker:latest <master_node_ip>:32000/exaflow_worker:latest
sudo docker tag madgik/exaflow_aggregation_server:latest <master_node_ip>:32000/exaflow_aggregation_server:latest
sudo docker tag gpikra/coordinator:v7.0.7.4 <master_node_ip>:32000/coordinator:v7.0.7.4
sudo docker tag mongo:5.0.8 <master_node_ip>:32000/mongo:5.0.8
sudo docker tag redis:alpine3.15 <master_node_ip>:32000/redis:alpine3.15
```

```
sudo docker push <master_node_ip>:32000/exaflow_controller:latest
sudo docker push <master_node_ip>:32000/exaflow_worker:latest
sudo docker push <master_node_ip>:32000/exaflow_aggregation_server:latest
sudo docker push <master_node_ip>:32000/coordinator:v7.0.7.4
sudo docker push <master_node_ip>:32000/mongo:5.0.8
sudo docker push <master_node_ip>:32000/redis:alpine3.15
```

**Attention:** Pulling, tagging and pushing the images in the private registry needs to be repeated every time there is a new image that needs to be deployed in the cluster.

### In ALL nodes follow these instructions:

Configure microk8s in **all nodes** to allow the use of the private registry following this guide:
https://microk8s.io/docs/registry-private

## Firewall Configuration

Using firewalld the following rules should apply,

in the **master** node, to expose the controller api, if it's used outside the cluster:

```
firewall-cmd --permanent --add-port=30000/tcp      # Exaflow Controller port
```

on all nodes:

```
sudo firewall-cmd --permanent --add-port={10255,12379,25000,16443,10250,10257,10259,32000}/tcp
sudo firewall-cmd --reload
```

If there are node to node communication problems we also need to add on all nodes:

```
SUBNET=`cat /var/snap/microk8s/current/args/cni-network/cni.yaml | grep CALICO_IPV4POOL_CIDR -a1 | tail -n1 | grep -oP '[\d\./]+'`
sudo firewall-cmd --permanent --new-zone=microk8s-cluster
sudo firewall-cmd --permanent --zone=microk8s-cluster --set-target=ACCEPT
sudo firewall-cmd --permanent --zone=microk8s-cluster --add-source=$SUBNET
sudo firewall-cmd --reload
```
