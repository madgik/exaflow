## Deploy Exaflow

1. Configure the [helm chart values](../values.yaml).

   - The `exaflow_images -> version` should be the exaflow services version in dockerhub.
   - Set `worker_count` to the number of Exaflow workers to deploy.

1. From the `Exaflow` folder, deploy the services:

```
helm install exaflow kubernetes
```

**For a deployment with microk8s use `microk8s helm3` in the commands.**

### Change the Exaflow version running

1. Modify the `exaflow_images -> version` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade exaflow kubernetes
```

### Increase/reduce the number of local nodes

1. Modify the `worker_count` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade exaflow kubernetes
```

### Restart the federation

You can restart the federation with helm by running:

```
helm uninstall exaflow
helm install exaflow kubernetes
```

## Log Rotation Configuration

In order to avoid docker logs taking up too much space in the **master**/**worker** nodes, the docker engine should be configured appropriately in every node.

[Docker-Engine Log Rotation Guide](https://docs.docker.com/config/containers/logging/configure/#configure-the-default-logging-driver)
