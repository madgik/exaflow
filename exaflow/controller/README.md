## CONTROLLER service

### Build docker image

To build a new image you must be on the project root `Exaflow`, then

```
docker build -t <USERNAME>/exaflow_controller:<IMAGETAG> -f exaflow/controller/Dockerfile .
```

## Run with env file

The controller reads configuration from `EXAFLOW_CONTROLLER_CONFIG_FILE` if set;
otherwise it loads `exaflow/controller/config.toml` and interpolates values
from environment variables. If you use an env file, provide all variables
referenced in `exaflow/controller/config.toml`.

Create a file with the workers' locations, for example:

```
["172.17.0.1:5670", "172.17.0.1:5671", "172.17.0.1:5672"]
```

Create an env_file with the following variables (adjust as needed):

```
NODE_IDENTIFIER=controller
FEDERATION=dementia
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
DEPLOYMENT_TYPE=LOCAL
WORKER_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL=30
WORKER_TASKS_TIMEOUT=20
GRPC_MAX_MESSAGE_MB=100
FLOWER_ENABLED=true
FLOWER_EXECUTION_TIMEOUT=30
FLOWER_SERVER_PORT=8080
LOCALWORKERS_CONFIG_FILE=/home/user/localworkers_config.json
LOCALWORKERS_DNS=
LOCALWORKERS_PORT=
AGGREGATION_SERVER_ENABLED=false
AGGREGATION_SERVER_DNS=
SMPC_ENABLED=false
SMPC_OPTIONAL=false
SMPC_COORDINATOR_ADDRESS=
SMPC_GET_RESULT_INTERVAL=10
SMPC_GET_RESULT_MAX_RETRIES=100
DP_ENABLED=false
DP_SENSITIVITY=
DP_PRIVACY_BUDGET=
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/exaflow_controller:<IMAGETAG>
```
