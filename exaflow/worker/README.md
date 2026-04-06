## WORKER service

### Build docker image

To build a new image you must be on the project root `Exaflow`, then

```
docker build -t <USERNAME>/exaflow_worker:<IMAGETAG> -f exaflow/worker/Dockerfile .
```

## Run with env file

The worker reads configuration from `EXAFLOW_WORKER_CONFIG_FILE` if set;
otherwise it loads `exaflow/worker/config.toml` and interpolates values from
environment variables. If you use an env file, provide all variables referenced
in `exaflow/worker/config.toml`.

Create an env_file with the following variables:

```
WORKER_IDENTIFIER=globalworker
WORKER_ROLE=GLOBALWORKER
FEDERATION=dementia
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
DUCKDB_PATH=/opt/data/data_models.duckdb
DATA_PATH=/opt/csvs/
CONTROLLER_IP=172.17.0.1
CONTROLLER_PORT=5000
WORKER_TASKS_TIMEOUT=20
PROTECT_LOCAL_DATA=false
GRPC_IP=172.17.0.1
GRPC_PORT=5670
GRPC_MAX_WORKERS=10
AGGREGATION_SERVER_ENABLED=false
AGGREGATION_SERVER_DNS=
SMPC_ENABLED=false
SMPC_OPTIONAL=false
SMPC_CLIENT_ID=
SMPC_CLIENT_ADDRESS=
SMPC_COORDINATOR_ADDRESS=
```

`DATA_PATH` should point at the directory containing the input CSVs. Set
`DUCKDB_PATH` to a writable location (for example `/opt/data/data_models.duckdb`)
when CSVs are mounted read-only.

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/exaflow_worker:<IMAGETAG>
```
