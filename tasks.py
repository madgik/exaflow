"""
Deployment script used for the development of the Exaflow.

In order to understand this script a basic knowledge of the system is required, this script
does not contain the documentation of the engine. The documentation in this script
is targeted to the specifics of the development deployment process.

This script deploys all the containers and api natively on your machine.
It deploys the containers on different ports and then configures the api to use the appropriate ports.

A worker service uses a configuration file either on the default location './exaflow/worker/config.toml'
or in the location of the env variable 'EXAFLOW_WORKER_CONFIG_FILE', if the env variable is set.
This deployment script used for development, uses the env variable logic, therefore before deploying each
worker service the env variable is changed to the location of the worker api' config file.

In order for this script to work the './configs/workers' folder should contain all the worker's config files
following the './exaflow/worker/config.toml' as template.
You can either create the files manually or using a '.deployment.toml' file with the following template
```
ip = "172.17.0.1"
log_level = "INFO"
framework_log_level ="INFO"

[controller]
port = 5100

[[workers]]
id = "globalworker"
grpc_port=5670

[[workers]]
id = "localworker1"
grpc_port=5671

[[workers]]
id = "localworker2"
grpc_port=5672
```

and by running the command 'inv create-configs'.

The worker api are named after their config file. If a config file is named './configs/workers/localworker1.toml'
the worker service will be called 'localworker1' and should be referenced using that in the following commands.

Paths are subject to change so in the following documentation the global variables will be used.

"""

import copy
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextlib import contextmanager
from enum import Enum
from itertools import cycle
from os import listdir
from os import path
from pathlib import Path
from textwrap import indent
from time import sleep

import requests
import toml
from invoke import UnexpectedExit
from invoke import task
from termcolor import colored

PROJECT_ROOT = Path(__file__).parent
DEPLOYMENT_CONFIG_FILE = PROJECT_ROOT / ".deployment.toml"
WORKERS_CONFIG_DIR = PROJECT_ROOT / "configs" / "workers"
WORKER_CONFIG_TEMPLATE_FILE = PROJECT_ROOT / "exaflow" / "worker" / "config.toml"
CONTROLLER_CONFIG_DIR = PROJECT_ROOT / "configs" / "controller"
AGG_SERVER_CONFIG_DIR = PROJECT_ROOT / "configs" / "aggregation_server"
CONTROLLER_LOCALWORKERS_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "controller" / "localworkers_config.json"
)
CONTROLLER_CONFIG_TEMPLATE_FILE = (
    PROJECT_ROOT / "exaflow" / "controller" / "config.toml"
)
AGG_SERVER_DIR = PROJECT_ROOT / "exaflow" / "aggregation_server"
AGG_SERVER_CONFIG_TEMPLATE_FILE = AGG_SERVER_DIR / "config.toml"
OUTDIR = Path("/tmp/exaflow/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

CLEANUP_DIR = Path("/tmp/cleanup_entries/")
if not CLEANUP_DIR.exists():
    CLEANUP_DIR.mkdir()

TEST_DATA_FOLDER = PROJECT_ROOT / "tests" / "test_data"

FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE = "FLOWER_ALGORITHM_FOLDERS"
EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAREME3_ALGORITHM_FOLDERS"
EXAFLOW_WORKER_CONFIG_FILE = "EXAFLOW_WORKER_CONFIG_FILE"
EXAFLOW_CONTROLLER_CONFIG_FILE = "EXAFLOW_CONTROLLER_CONFIG_FILE"
EXAFLOW_AGG_SERVER_CONFIG_FILE = "EXAFLOW_AGG_SERVER_CONFIG_FILE"
DATA_PATH = "DATA_PATH"
WORKER_STARTUP_SUCCESS_LOG = "Data folder loaded successfully on startup."
WORKER_STARTUP_TIMEOUT_SECONDS = 120


def _expand_path(path_value) -> Path:
    """Expand environment variables and user symbols inside *path_value*."""

    return Path(os.path.expanduser(os.path.expandvars(str(path_value))))


def _load_worker_config(worker_id: str) -> dict | None:
    config_path = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
    if not config_path.exists():
        return None
    with open(config_path) as fp:
        return toml.load(fp)


def _worker_data_path(worker_id: str) -> Path:
    return TEST_DATA_FOLDER / ".data_paths" / worker_id


SMPC_COORDINATOR_PORT = 12314
SMPC_COORDINATOR_DB_PORT = 27017
SMPC_COORDINATOR_QUEUE_PORT = 6379
SMPC_PLAYER_BASE_PORT = 7000
SMPC_CLIENT_BASE_PORT = 9000
SMPC_COORDINATOR_NAME = "smpc_coordinator"
SMPC_COORDINATOR_DB_NAME = "smpc_coordinator_db"
SMPC_COORDINATOR_QUEUE_NAME = "smpc_coordinator_queue"
SMPC_PLAYER_BASE_NAME = "smpc_player"
SMPC_CLIENT_BASE_NAME = "smpc_client"


# TODO Replace invoke's lack of pre-task support when implemented https://github.com/pyinvoke/invoke/issues/170
# Right now if we call a task from another task, the "pre"-task is not executed


@task
def create_configs(c):
    """
    Create the worker and controller api config files, using 'DEPLOYMENT_CONFIG_FILE'.
    """
    if path.exists(WORKERS_CONFIG_DIR) and path.isdir(WORKERS_CONFIG_DIR):
        shutil.rmtree(WORKERS_CONFIG_DIR)
    WORKERS_CONFIG_DIR.mkdir(parents=True)

    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Deployment config file '{DEPLOYMENT_CONFIG_FILE}' not found."
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        deployment_config = toml.load(fp)

    with open(WORKER_CONFIG_TEMPLATE_FILE) as fp:
        template_worker_config = toml.load(fp)

    for worker in deployment_config["workers"]:
        worker_config = copy.deepcopy(template_worker_config)

        aggregation_enabled = deployment_config["aggregation_server"]["enabled"]
        aggregation_dns = ""
        if aggregation_enabled:
            aggregation_dns = f"{deployment_config['ip']}:{deployment_config['aggregation_server']['port']}"

        worker_config["identifier"] = worker["id"]
        worker_config["role"] = worker["role"]
        worker_config["federation"] = deployment_config["federation"]
        worker_config["log_level"] = deployment_config["log_level"]
        worker_config["framework_log_level"] = deployment_config["framework_log_level"]
        worker_config["controller"]["ip"] = deployment_config["ip"]
        worker_config["controller"]["port"] = deployment_config["controller"]["port"]
        worker_config["aggregation_server"]["enabled"] = aggregation_enabled
        worker_config["aggregation_server"]["dns"] = aggregation_dns

        worker_config["grpc"]["ip"] = deployment_config["ip"]
        worker_config["grpc"]["port"] = worker["grpc_port"]
        worker_config["grpc"]["bind_ip"] = "0.0.0.0"

        worker_config["worker_tasks"]["tasks_timeout"] = deployment_config[
            "worker_tasks_timeout"
        ]

        worker_config["privacy"]["minimum_row_count"] = deployment_config["privacy"][
            "minimum_row_count"
        ]
        if worker["role"] == "GLOBALWORKER":
            worker_config["privacy"]["protect_local_data"] = False
        else:
            worker_config["privacy"]["protect_local_data"] = deployment_config[
                "privacy"
            ]["protect_local_data"]

        worker_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
        if worker_config["smpc"]["enabled"]:
            worker_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]
            if coordinator_ip := deployment_config["smpc"].get("coordinator_ip"):
                if worker["role"] == "GLOBALWORKER":
                    worker_config["smpc"]["coordinator_address"] = (
                        f"http://{coordinator_ip}:{SMPC_COORDINATOR_PORT}"
                    )
                else:
                    worker_config["smpc"]["client_id"] = worker["smpc_client_id"]
                    worker_config["smpc"]["client_address"] = (
                        f"http://{coordinator_ip}:{worker['smpc_client_port']}"
                    )
            else:
                if worker["role"] == "GLOBALWORKER":
                    worker_config["smpc"]["coordinator_address"] = (
                        f"http://{deployment_config['ip']}:{SMPC_COORDINATOR_PORT}"
                    )
                else:
                    worker_config["smpc"]["client_id"] = worker["id"]
                    worker_config["smpc"]["client_address"] = (
                        f"http://{deployment_config['ip']}:{worker['smpc_client_port']}"
                    )

        worker_config["duckdb"] = {
            "path": f"{str(_worker_data_path(worker_config['identifier']))}/data_models.duckdb"
        }

        worker_config_file = WORKERS_CONFIG_DIR / f"{worker['id']}.toml"
        with open(worker_config_file, "w+") as fp:
            toml.dump(worker_config, fp)

    # Create the controller config file
    with open(CONTROLLER_CONFIG_TEMPLATE_FILE) as fp:
        template_controller_config = toml.load(fp)
    controller_config = copy.deepcopy(template_controller_config)
    controller_config["node_identifier"] = "controller"
    controller_config["federation"] = deployment_config["federation"]
    controller_config["log_level"] = deployment_config["log_level"]
    controller_config["framework_log_level"] = deployment_config["framework_log_level"]

    controller_config["worker_landscape_aggregator_update_interval"] = (
        deployment_config["worker_landscape_aggregator_update_interval"]
    )
    controller_config["flower"]["enabled"] = deployment_config["flower"]["enabled"]
    controller_config["flower"]["execution_timeout"] = deployment_config["flower"][
        "execution_timeout"
    ]
    controller_config["flower"]["server_port"] = deployment_config["flower"][
        "server_port"
    ]
    aggregation_enabled = deployment_config["aggregation_server"]["enabled"]
    controller_config["aggregation_server"]["enabled"] = aggregation_enabled
    controller_config["aggregation_server"]["dns"] = (
        f"{deployment_config['ip']}:{deployment_config['aggregation_server']['port']}"
        if aggregation_enabled
        else ""
    )
    controller_config["worker_tasks_timeout"] = deployment_config[
        "worker_tasks_timeout"
    ]

    controller_config["deployment_type"] = "LOCAL"

    controller_config["localworkers"]["config_file"] = str(
        CONTROLLER_LOCALWORKERS_CONFIG_FILE
    )
    controller_config["localworkers"]["dns"] = ""
    controller_config["localworkers"]["port"] = ""

    controller_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
    if controller_config["smpc"]["enabled"]:
        controller_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]
        if coordinator_ip := deployment_config["smpc"].get("coordinator_ip"):
            controller_config["smpc"]["coordinator_address"] = (
                f"http://{coordinator_ip}:{SMPC_COORDINATOR_PORT}"
            )
        else:
            controller_config["smpc"]["coordinator_address"] = (
                f"http://{deployment_config['ip']}:{SMPC_COORDINATOR_PORT}"
            )

        controller_config["smpc"]["get_result_interval"] = deployment_config["smpc"][
            "get_result_interval"
        ]
        controller_config["smpc"]["get_result_max_retries"] = deployment_config["smpc"][
            "get_result_max_retries"
        ]

        controller_config["smpc"]["dp"]["enabled"] = deployment_config["smpc"]["dp"][
            "enabled"
        ]
        if controller_config["smpc"]["dp"]["enabled"]:
            controller_config["smpc"]["dp"]["sensitivity"] = deployment_config["smpc"][
                "dp"
            ]["sensitivity"]
            controller_config["smpc"]["dp"]["privacy_budget"] = deployment_config[
                "smpc"
            ]["dp"]["privacy_budget"]

    CONTROLLER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"
    with open(controller_config_file, "w+") as fp:
        toml.dump(controller_config, fp)

    # Create the controller localworkers config file
    localworkers_addresses = [
        f"{deployment_config['ip']}:{worker['grpc_port']}"
        for worker in deployment_config["workers"]
    ]
    with open(CONTROLLER_LOCALWORKERS_CONFIG_FILE, "w+") as fp:
        json.dump(localworkers_addresses, fp)

    # Create the aggregation_server config file
    if not deployment_config["aggregation_server"]["enabled"]:
        return
    with open(AGG_SERVER_CONFIG_TEMPLATE_FILE) as fp:
        template_aggregation_server_config = toml.load(fp)
    aggregation_server_config = copy.deepcopy(template_aggregation_server_config)
    aggregation_server_config["enabled"] = deployment_config["aggregation_server"][
        "enabled"
    ]
    aggregation_server_config["port"] = deployment_config["aggregation_server"]["port"]
    aggregation_server_config["max_grpc_connections"] = deployment_config[
        "aggregation_server"
    ]["max_grpc_connections"]
    aggregation_server_config["max_wait_for_aggregation_inputs"] = deployment_config[
        "aggregation_server"
    ]["max_wait_for_aggregation_inputs"]
    aggregation_server_config["log_level"] = deployment_config["log_level"]

    AGG_SERVER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    aggregation_server_config_file = AGG_SERVER_CONFIG_DIR / "aggregation_server.toml"
    with open(aggregation_server_config_file, "w+") as fp:
        toml.dump(aggregation_server_config, fp)


@task
def install_dependencies(c):
    """Install project dependencies using poetry."""
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    run(c, cmd)


@task
def rm_containers(c, container_name=None, smpc=False):
    """
    Remove the specified docker containers, either by container or relative name.

    :param container_name: If set, removes the container with the specified name.
    :param smpc: If True, it will remove all smpc related containers.

    If nothing is set, nothing is removed.
    """
    names = []
    if smpc:
        names.append("smpc")
    if container_name:
        names.append(container_name)
    if not names:
        message(
            "You must specify a container family to remove",
            level=Level.WARNING,
        )
    for name in names:
        container_ids = run(c, f"docker ps -qa --filter name={name}", show_ok=False)
        if container_ids.stdout:
            message(f"Removing {name} container(s)...", Level.HEADER)
            cmd = f"docker rm -vf $(docker ps -qa --filter name={name})"
            run(c, cmd)
        else:
            message(f"No {name} container to remove.", level=Level.HEADER)


@task(iterable=["worker"], name="create-duckdb")
def create_duckdb(c, worker):
    """Prepare DuckDB files for the specified workers."""

    if not worker:
        message("Please specify a worker using --worker <worker>", Level.WARNING)
        sys.exit(1)

    for worker_id in worker:
        duckdb_path = _worker_data_path(worker_id)
        clean_duckdb(duckdb_path)
        message(
            f"Reset DuckDB file for worker: {worker_id} at {duckdb_path}.",
            Level.HEADER,
        )


@task
def update_wla(c):
    url = f"http://localhost:{get_controller_port()}/wla"
    try:
        response = requests.post(url, timeout=10)
    except requests.RequestException as exc:
        message(
            f"Warning: failed to update wla ({exc}); controller may still be starting.",
            Level.WARNING,
        )
        return

    if response.status_code != 200:
        raise Exception("Failed to update the wla")
    print("Successfully updated wla.")


def _structure_data(worker=None):
    """
    Delegate dataset structuring to worker_data_paths_builder.py and
    reprint its output using our message() logging system.
    """

    def get_worker_configs():
        config_files = [
            WORKERS_CONFIG_DIR / file for file in listdir(WORKERS_CONFIG_DIR)
        ]
        if not config_files:
            raise RuntimeError(f"No worker configs in: {WORKERS_CONFIG_DIR}")

        configs = []
        for cfg in config_files:
            with open(cfg) as fp:
                configs.append(toml.load(fp))
        return configs

    def filter_worker_configs(worker_configs, worker_identifier, node_type):
        return [
            config["identifier"]
            for config in worker_configs
            if (not worker_identifier or config["identifier"] == worker_identifier)
            and config["role"] == node_type
        ]

    worker_configs = get_worker_configs()

    local_worker_ids = filter_worker_configs(worker_configs, worker, "LOCALWORKER")
    if not local_worker_ids:
        raise Exception("No local workers found.")

    script_path = Path(__file__).parent / "tests" / "worker_data_paths_builder.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--local-workers",
        *local_worker_ids,
    ]

    # Run helper script and capture its output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Reprint script output inside Removing directoryyour message() logger
    if result.stdout.strip():
        message(result.stdout.strip(), Level.HEADER)

    if result.stderr.strip():
        message(result.stderr.strip(), Level.ERROR)

    # Raise if failed
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


@task(iterable=["port"])
def structure_data(c, worker=None):
    _structure_data(worker)


@task
def kill_worker(c, worker=None, all_=False, silence=False):
    """
    Kill the worker(s) service(s).

    :param worker: The worker service to kill.
    :param all_: If set, all worker api will be killed.
    """

    if all_:
        worker_pattern = ""
    elif worker:
        worker_pattern = worker
    else:
        message(
            "Please specify a worker using --worker <worker> or use --all",
            Level.WARNING,
        )
        sys.exit(1)

    process_pattern = "python -m exaflow.worker.grpc_server"
    grep_cmd = f"ps aux | grep '[p]ython -m exaflow.worker.grpc_server' | grep '{worker_pattern}' "
    res_bin = run(c, grep_cmd, warn=True, show_ok=False)

    if res_bin.ok:
        if not silence:
            message(
                f"Killing previous worker gRPC instance(s) with pattern '{worker_pattern}' ...",
                Level.HEADER,
            )
        cmd = (
            f"pids=$(ps aux | grep '[p]ython -m exaflow.worker.grpc_server' | grep '{worker_pattern}' | awk '{{print $2}}') "
            '&& if [ -n "$pids" ]; then kill -9 $pids; fi'
        )
        run(c, cmd, warn=True, show_ok=False)
    else:
        if not silence:
            message("No worker instances found", Level.HEADER)


def validate_algorithm_folders(folders, name):
    """Validates and retrieves the algorithm folder configuration."""
    if not folders:
        folders = get_deployment_config(name)
    if not isinstance(folders, str):
        raise ValueError(f"The {name} configuration must be a comma-separated string.")
    return folders


@contextmanager
def env_prefixes(c, env_vars):
    """
    A context manager that applies a set of environment variable prefixes
    using an ExitStack.
    """
    with ExitStack() as stack:
        for key, value in env_vars.items():
            stack.enter_context(c.prefix(f"export {key}={value}"))
        yield


def _wait_for_worker_startup(worker_id: str, log_path: Path):
    """Block until the worker log contains the startup success message."""
    deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if log_path.exists():
            try:
                with log_path.open() as fp:
                    log_contents = fp.read()
            except OSError as exc:
                message(
                    f"Could not read log file for worker {worker_id}: {exc}",
                    Level.WARNING,
                )
            else:
                if WORKER_STARTUP_SUCCESS_LOG in log_contents:
                    message(f"Worker {worker_id} ready.", Level.SUCCESS)
                    return
        sleep(1)
    raise TimeoutError(
        f"Worker {worker_id} did not log '{WORKER_STARTUP_SUCCESS_LOG}' within "
        f"{WORKER_STARTUP_TIMEOUT_SECONDS} seconds. Check logs at {log_path}."
    )


@task
def start_worker(
    c,
    worker=None,
    all_=False,
    framework_log_level=None,
    detached=False,
    flower_algorithm_folders=None,
    exareme3_algorithm_folders=None,
):
    """
    (Re)Start the worker(s) service(s). If a worker service is running, stop and start it again.

    :param worker: The worker to start, using the proper file in the `WORKERS_CONFIG_DIR`.
    :param all_: If set, the workers of which the configuration file exists, will be started.
    :param framework_log_level: If not provided, it will look into the `DEPLOYMENT_CONFIG_FILE`.
    :param detached: If set to True, it will start the service in the background.
    :param flower_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.

    The containers related to the api remain unchanged.
    """

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    # Validate algorithm folders
    flower_algorithm_folders = validate_algorithm_folders(
        flower_algorithm_folders, "flower_algorithm_folders"
    )
    exareme3_algorithm_folders = validate_algorithm_folders(
        exareme3_algorithm_folders, "exareme3_algorithm_folders"
    )

    worker_ids = sorted(get_worker_ids(all_, worker))

    # If we're not detached and not starting all, keep the old sequential / attached behavior
    if len(worker_ids) == 1:
        worker_id = worker_ids[0]
        kill_worker(c, worker_id)
        message(f"Starting Worker {worker_id}...", Level.HEADER)
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"

        env_vars = {
            FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE: flower_algorithm_folders,
            EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE: exareme3_algorithm_folders,
            EXAFLOW_WORKER_CONFIG_FILE: worker_config_file,
            DATA_PATH: (_worker_data_path(worker_id)).as_posix(),
        }

        with env_prefixes(c, env_vars):
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT} poetry run python -m "
                f"exaflow.worker.grpc_server --worker-id {worker_id}"
            )
            run(c, cmd, attach_=True)
        return  # we're done

    # ------------------------------------------------------------------
    # Parallel branch: detached / all_ (background workers)
    # ------------------------------------------------------------------

    def _restart_single_worker(worker_id: str):
        """Kill and start a single worker (background)."""
        kill_worker(c, worker_id, silence=True)

        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        env_vars = {
            FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE: flower_algorithm_folders,
            EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE: exareme3_algorithm_folders,
            EXAFLOW_WORKER_CONFIG_FILE: worker_config_file,
            DATA_PATH: (_worker_data_path(worker_id)).as_posix(),
        }

        with env_prefixes(c, env_vars):
            outpath = OUTDIR / f"{worker_id}.out"
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT}: poetry run python -m "
                f"exaflow.worker.grpc_server --worker-id {worker_id} "
                f"> {outpath} 2>&1"
            )
            run(c, cmd, wait=False)
        _wait_for_worker_startup(worker_id, outpath)

    message(f"Starting Workers {worker_ids}...", Level.HEADER)
    # Use as many threads as workers (you can cap this if you like)
    max_workers = max(1, len(worker_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all workers in parallel
        futures = [executor.submit(_restart_single_worker, wid) for wid in worker_ids]
        # Block until every worker logs the expected startup message
        for future in futures:
            future.result()


@task
def kill_aggregation_server(c):
    """
    Kill any running exaflow.aggregation_server.server processes.
    """
    res = c.run(
        "ps aux | grep '[e]xaflow.aggregation_server.server'", warn=True, hide="both"
    )
    if res.ok and res.stdout.strip():
        message("Killing existing aggregation_server ...", Level.HEADER)
        c.run(
            "ps aux | grep '[e]xaflow.aggregation_server.server' "
            "| awk '{print $2}' | xargs kill -9",
            warn=True,
        )
        message("Ok", Level.SUCCESS)
    else:
        message("No aggregation_server process found.", Level.HEADER)


@task(pre=[kill_aggregation_server])
def start_aggregation_server(c, detached: bool = False):
    """Start the aggregation_server gRPC service."""
    kill_aggregation_server(c)
    message("Starting aggregation server...", Level.HEADER)

    config_file = AGG_SERVER_CONFIG_DIR / "aggregation_server.toml"
    if not config_file.exists():
        config_file = AGG_SERVER_CONFIG_TEMPLATE_FILE
    if not config_file.exists():
        message(f"Config not found: {config_file}", Level.ERROR)
        return

    env = os.environ.copy()
    env["AGG_SERVER_CONFIG_FILE"] = str(config_file)

    # run the script directly instead of -m exaflow.aggregation_server.server
    run_cmd = (
        f"cd {PROJECT_ROOT!s} && poetry run python -m exaflow.aggregation_server.server"
    )

    if detached:
        logf = OUTDIR / "aggregation_server.out"
        c.run(
            f"{run_cmd} >> {logf!s} 2>&1 &",
            env=env,
            pty=False,
            disown=True,
            warn=True,
        )
        message("Ok.", Level.SUCCESS)
    else:
        c.run(run_cmd, env=env, pty=True)


@task
def kill_controller(c):
    """Kill the controller service."""
    controller_port = get_controller_port()
    lsof_res = run(
        c,
        f"lsof -nP -iTCP:{controller_port} -sTCP:LISTEN",
        warn=True,
        show_ok=False,
    )
    pids_to_kill = set()
    non_controller_listeners = []
    if lsof_res.ok and lsof_res.stdout.strip():
        for line in lsof_res.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            command = parts[0].lower()
            pid = parts[1]
            if command in ("python", "hypercorn"):
                pids_to_kill.add(pid)
            else:
                non_controller_listeners.append((parts[0], pid))

    if pids_to_kill:
        message("Killing previous Hypercorn instances...", Level.HEADER)
        pids = " ".join(sorted(pids_to_kill))
        cmd = f"kill -9 {pids} && sleep 1"
        run(c, cmd, warn=True)
        return

    # Fallback for controllers that may not expose a listening socket yet.
    hypercorn_process_patterns = (
        "hypercorn --config python:exaflow.controller.quart.hypercorn_config",
        "exaflow.controller.quart.app:app",
    )
    for process_pattern in hypercorn_process_patterns:
        res = run(c, f"pgrep -f '{process_pattern}'", warn=True, show_ok=False)
        if res.ok and res.stdout.strip():
            pids = " ".join(res.stdout.strip().splitlines())
            message("Killing previous Hypercorn instances...", Level.HEADER)
            cmd = f"kill -9 {pids} && sleep 1"
            run(c, cmd, warn=True)
            return

    if non_controller_listeners:
        listeners = ", ".join(
            f"{command}(pid={pid})" for command, pid in non_controller_listeners
        )
        message(
            f"No controller process found. Port {controller_port} is used by: {listeners}",
            Level.WARNING,
        )
        return

    message("No hypercorn instance found", Level.HEADER)


@task
def start_controller(
    c,
    detached=False,
    flower_algorithm_folders=None,
    exareme3_algorithm_folders=None,
):
    """
    (Re)Start the controller service. If the service is already running, stop and start it again.
    """

    # Validate algorithm folders
    flower_algorithm_folders = validate_algorithm_folders(
        flower_algorithm_folders, "flower_algorithm_folders"
    )
    exareme3_algorithm_folders = validate_algorithm_folders(
        exareme3_algorithm_folders, "exareme3_algorithm_folders"
    )

    kill_controller(c)
    message("Starting Controller...", Level.HEADER)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"
    controller_port = get_controller_port()

    # Build a dictionary of environment variables for the controller
    env_vars = {
        FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE: flower_algorithm_folders,
        EXAREME3_ALGORITHM_FOLDERS_ENV_VARIABLE: exareme3_algorithm_folders,
        EXAFLOW_CONTROLLER_CONFIG_FILE: controller_config_file,
    }

    # Use the helper context manager to apply environment variable prefixes
    with env_prefixes(c, env_vars):
        outpath = OUTDIR / "controller.out"
        if detached:
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT} poetry run hypercorn --config python:exaflow.controller.quart.hypercorn_config "
                f"-b 0.0.0.0:{controller_port} exaflow.controller.quart.app:app >> {outpath} 2>&1"
            )
            run(c, cmd, wait=False)
        else:
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT} poetry run hypercorn --config python:exaflow.controller.quart.hypercorn_config "
                f"-b 0.0.0.0:{controller_port} exaflow.controller.quart.app:app"
            )
            run(c, cmd, attach_=True)


@task
def deploy(
    c,
    install_dep=True,
    log_level=None,
    framework_log_level=None,
    flower_algorithm_folders=None,
    exareme3_algorithm_folders=None,
    smpc=None,
    no_structure_data=False,
):
    """
    Install dependencies, (re)create all the containers and (re)start all the api.

    :param install_dep: Install dependencies or not.
    :param log_level: Used for the dev logs. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param framework_log_level: Used for the engine api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param flower_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param exareme3_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc: Deploy the SMPC cluster as well. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param no_structure_data: Do not structure data on the folder `tests/test_data/.data_paths`.
    """

    if not log_level:
        log_level = get_deployment_config("log_level")

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    if not flower_algorithm_folders:
        flower_algorithm_folders = get_deployment_config("flower_algorithm_folders")

    if not exareme3_algorithm_folders:
        exareme3_algorithm_folders = get_deployment_config("exareme3_algorithm_folders")

    start_aggregation_server_ = get_deployment_config(
        "aggregation_server", subconfig="enabled"
    )

    if smpc is None:
        smpc = get_deployment_config("smpc", subconfig="enabled")

    if install_dep:
        install_dependencies(c)

    # Start WORKER api
    config_files = [WORKERS_CONFIG_DIR / file for file in listdir(WORKERS_CONFIG_DIR)]
    if not config_files:
        message(
            f"There are no worker config files to be used for deployment. Folder: {WORKERS_CONFIG_DIR}",
            Level.WARNING,
        )
        sys.exit(1)

    worker_ids = []
    for worker_config_file in config_files:
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        worker_ids.append(worker_config["identifier"])

    if not no_structure_data:
        _structure_data()

    worker_ids.sort()  # Sorting the ids protects removing a similarly named id, localworker1 would remove localworker10.
    if start_aggregation_server_:
        start_aggregation_server(c, detached=True)

    start_worker(
        c,
        all_=True,
        framework_log_level=framework_log_level,
        detached=True,
        flower_algorithm_folders=flower_algorithm_folders,
        exareme3_algorithm_folders=exareme3_algorithm_folders,
    )

    # Start CONTROLLER service
    start_controller(
        c,
        detached=True,
        flower_algorithm_folders=flower_algorithm_folders,
        exareme3_algorithm_folders=exareme3_algorithm_folders,
    )

    if smpc and not get_deployment_config("smpc", subconfig="coordinator_ip"):
        deploy_smpc(c)


@task
def attach(c, worker=None, controller=False, db=None):
    """
    Attach to a worker/controller service or a db container.

    :param worker: The worker service name to which to attach.
    :param controller: Attach to controller flag.
    :param db: The db container name to which to attach.
    """
    if (worker or controller) and not (worker and controller):
        fname = worker or "controller"
        outpath = OUTDIR / (fname + ".out")
        cmd = f"tail -f {outpath}"
        run(c, cmd, attach_=True)
    elif db:
        run(c, f"docker exec -it {db} mclient db", attach_=True)
    else:
        message("You must attach to Worker, Controller or DB", Level.WARNING)
        sys.exit(1)


def _on_rm_error(func, path, exc_info):
    """
    If removing a file/dir fails due to permissions, chmod it and retry.
    """
    exc = exc_info[1]
    if isinstance(exc, PermissionError):
        try:
            os.chmod(path, stat.S_IWUSR | stat.S_IREAD | stat.S_IEXEC)
            func(path)
        except Exception:
            raise
    else:
        raise exc


@task
def cleanup(c):
    """Stop worker/controller services and delete DuckDB files."""
    kill_controller(c)
    kill_aggregation_server(c)
    kill_worker(c, all_=True)
    rm_containers(c, smpc=True)

    # Remove any DuckDB files referenced in worker configs (in case they live elsewhere)
    if WORKERS_CONFIG_DIR.exists():
        config_defined_paths = set()
        for config_file in WORKERS_CONFIG_DIR.glob("*.toml"):
            with open(config_file) as fp:
                worker_config = toml.load(fp)
            duckdb_path = worker_config.get("duckdb", {}).get("path")
            if duckdb_path:
                config_defined_paths.add(_expand_path(duckdb_path))

    clean_data_paths()
    if OUTDIR.exists():
        message(f"Removing {OUTDIR}...", level=Level.HEADER)
        for outpath in OUTDIR.glob("*.out"):
            outpath.unlink()
        OUTDIR.rmdir()
        message("Ok", level=Level.SUCCESS)

    if CLEANUP_DIR.exists():
        message(f"Removing {CLEANUP_DIR}...", level=Level.HEADER)
        for cleanup_file in CLEANUP_DIR.glob("*.toml"):
            cleanup_file.unlink()
        CLEANUP_DIR.rmdir()
        message("Ok", level=Level.SUCCESS)


def clean_data_paths():
    combined_root_path = TEST_DATA_FOLDER / ".data_paths"

    if combined_root_path.exists():
        # Recursively find *all* .duckdb files
        for duckdb_path in combined_root_path.rglob("*.duckdb"):
            clean_duckdb(duckdb_path)

        try:
            message(
                f"Removing directory {combined_root_path} recursively...",
                level=Level.HEADER,
            )
            shutil.rmtree(combined_root_path, onerror=_on_rm_error)
            message("Ok", level=Level.SUCCESS)
        except Exception as exc:
            message(
                f"Failed to remove {combined_root_path}: {exc}",
                Level.WARNING,
            )


def clean_duckdb(duckdb_path):
    duckdb_path = _expand_path(duckdb_path)
    try:
        if duckdb_path.exists():
            message(f"Removing {duckdb_path}...", level=Level.HEADER)
            duckdb_path.unlink()
            message("Ok", level=Level.SUCCESS)
    except Exception as e:  # noqa: BLE001
        print(f"Error deleting {duckdb_path}: {e}")


def start_smpc_coordinator_db(c, image):
    container_ports = f"{SMPC_COORDINATOR_DB_PORT}:27017"
    message(
        f"Starting container {SMPC_COORDINATOR_DB_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    env_variables = (
        "-e MONGO_INITDB_ROOT_USERNAME=sysadmin -e MONGO_INITDB_ROOT_PASSWORD=123qwe "
    )
    cmd = f"docker run -d -p {container_ports} {env_variables} --name {SMPC_COORDINATOR_DB_NAME} {image}"
    run(c, cmd)


def start_smpc_coordinator_queue(c, image):
    container_ports = f"{SMPC_COORDINATOR_QUEUE_PORT}:6379"
    message(
        f"Starting container {SMPC_COORDINATOR_QUEUE_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    container_cmd = "redis-server --requirepass agora"
    cmd = f"""docker run -d -p {container_ports} -e REDIS_REPLICATION_MODE=master --name {SMPC_COORDINATOR_QUEUE_NAME} {image} {container_cmd}"""
    run(c, cmd)


def start_smpc_coordinator_container(c, ip, image):
    container_ports = f"{SMPC_COORDINATOR_PORT}:12314"
    message(
        f"Starting container {SMPC_COORDINATOR_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    container_cmd = "python coordinator.py"
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e REDIS_HOST={ip} "
        f"-e REDIS_PORT={SMPC_COORDINATOR_QUEUE_PORT} "
        "-e REDIS_PSWD=agora "
        f"-e DB_URL={ip}:{SMPC_COORDINATOR_DB_PORT} "
        "-e DB_UNAME=sysadmin "
        "-e DB_PSWD=123qwe "
    )
    cmd = f"""docker run -d -p {container_ports} {env_variables} --name {SMPC_COORDINATOR_NAME} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_coordinator(
    c, ip=None, smpc_image=None, smpc_db_image=None, smpc_queue_image=None
):
    """
    (Re)Creates all needed SMPC coordinator containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_image: The coordinator image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_db_image: The db image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_queue_image: The queue image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not smpc_image:
        smpc_image = get_deployment_config("smpc", subconfig="smpc_image")
    if not smpc_db_image:
        smpc_db_image = get_deployment_config("smpc", subconfig="db_image")
    if not smpc_queue_image:
        smpc_queue_image = get_deployment_config("smpc", subconfig="queue_image")

    get_docker_image(c, smpc_image)
    get_docker_image(c, smpc_db_image)
    get_docker_image(c, smpc_queue_image)

    rm_containers(c, container_name="smpc_coordinator")

    start_smpc_coordinator_db(c, smpc_db_image)
    start_smpc_coordinator_queue(c, smpc_queue_image)
    start_smpc_coordinator_container(c, ip, smpc_image)


def start_smpc_player(c, ip, id, image):
    name = f"{SMPC_PLAYER_BASE_NAME}_{id}"
    message(
        f"Starting container {name} ...",
        Level.HEADER,
    )
    container_cmd = f"python player.py {id}"  # SMPC player id cannot be alphanumeric
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e COORDINATOR_URL=http://{ip}:{SMPC_COORDINATOR_PORT} "
        f"-e DB_URL={ip}:{SMPC_COORDINATOR_DB_PORT} "
        "-e DB_UNAME=sysadmin "
        "-e DB_PSWD=123qwe "
        f"-e PORT={SMPC_PLAYER_BASE_PORT + id}"
    )
    container_ports = (
        f"-p {6000 + id}:{6000 + id} "
        f"-p {SMPC_PLAYER_BASE_PORT + id}:{7000 + id} "
        f"-p {14000 + id}:{14000 + id} "
    )  # SMPC player port is increasing using the player id
    cmd = f"""docker run -d {container_ports} {env_variables} --name {name} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_players(c, ip=None, image=None):
    """
    (Re)Creates 3 SMPC player containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param image: The smpc player image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not image:
        image = get_deployment_config("smpc", subconfig="smpc_image")

    get_docker_image(c, image)

    rm_containers(c, container_name="smpc_player")

    for i in range(3):
        start_smpc_player(c, ip, i, image)


def start_smpc_client(c, worker_id, ip, image):
    worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
    with open(worker_config_file) as fp:
        worker_config = toml.load(fp)

    client_id = worker_config["smpc"]["client_id"]
    client_port = worker_config["smpc"]["client_address"].split(":")[
        2
    ]  # Get the port from the address e.g. 'http://172.17.0.1:9000'

    name = f"{SMPC_CLIENT_BASE_NAME}_{client_id}"
    message(
        f"Starting container {name} ...",
        Level.HEADER,
    )
    container_cmd = f"python client.py"
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e COORDINATOR_URL=http://{ip}:{SMPC_COORDINATOR_PORT} "
        f"-e ID={client_id} "
        f"-e PORT={client_port} "
    )
    container_ports = f"-p {client_port}:{client_port} "
    cmd = f"""docker run -d {container_ports} {env_variables} --name {name} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_clients(c, ip=None, image=None):
    """
    (Re)Creates 3 SMPC player containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param image: The smpc player image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not image:
        image = get_deployment_config("smpc", subconfig="smpc_image")

    get_docker_image(c, image)

    rm_containers(c, container_name="smpc_client")

    for worker_id in get_localworker_ids():
        start_smpc_client(c, worker_id, ip, image)


@task
def deploy_smpc(c, ip=None, smpc_image=None, smpc_db_image=None, smpc_queue_image=None):
    """
    (Re)Creates all needed SMPC containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_image: The coordinator image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_db_image: The db image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_queue_image: The queue image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """
    rm_containers(c, smpc=True)
    start_smpc_coordinator(c, ip, smpc_image, smpc_db_image, smpc_queue_image)
    start_smpc_players(c, ip, smpc_image)
    start_smpc_clients(c, ip, smpc_image)


def run(
    c,
    cmd,
    attach_=False,
    wait=True,
    warn=False,
    raise_error=False,
    show_ok=True,
    env=None,
):
    if attach_:
        c.run(cmd, pty=True, env=env)
        return

    if not wait:
        # TODO disown=True will make c.run(..) return immediately
        c.run(cmd, disown=True, env=env)
        # TODO wait is False to get in here
        # nevertheless, it will wait (sleep) for 4 seconds here, why??
        spin_wheel(time=4)
        if show_ok:
            message("Ok", Level.SUCCESS)
        return

    # TODO this is supposed to run when wait=True, yet asynchronous=True
    promise = c.run(cmd, asynchronous=True, warn=warn, env=env)
    # TODO and then it blocks here, what is the point of asynchronous=True?
    spin_wheel(promise=promise)
    stderr = promise.runner.stderr
    if stderr and raise_error:
        raise UnexpectedExit(stderr)
    result = promise.join()
    if show_ok:
        message("Ok", Level.SUCCESS)
    return result


class Level(Enum):
    HEADER = ("cyan", 1)
    BODY = ("white", 2)
    SUCCESS = ("green", 1)
    ERROR = ("red", 1)
    WARNING = ("yellow", 1)


def message(msg, level=Level.BODY):
    if msg.endswith("..."):
        end = ""
    else:
        end = "\n"
    color, indent_level = level.value
    prfx = "  " * indent_level
    msg = colored(msg, color)
    print(indent(msg, prfx), end=end, flush=True)


wheel = cycle(r"-\|/")


def spin_wheel(promise=None, time=None):
    dt = 0.15
    if promise is not None:
        print("  ", end="")
        for frame in wheel:
            print(frame + "  ", sep="", end="", flush=True)
            sleep(dt)
            print("\b\b\b", sep="", end="", flush=True)
            if promise.runner.process_is_finished:
                print("\b\b", end="")
                break
    elif time:
        print("  ", end="")
        for frame in wheel:
            print(frame + "  ", sep="", end="", flush=True)
            sleep(dt)
            print("\b\b\b", sep="", end="", flush=True)
            time -= dt
            if time <= 0:
                print("\b\b", end="")
                break


def get_deployment_config(config, subconfig=None):
    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Please provide a --{config} parameter or create a deployment config file '{DEPLOYMENT_CONFIG_FILE}'"
        )
    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        if subconfig:
            return toml.load(fp)[config][subconfig]
        return toml.load(fp)[config]


def get_controller_port(default=5000):
    try:
        return int(get_deployment_config("controller", "port"))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return default


def get_worker_ids(all_=False, worker=None):
    worker_ids = []
    if all_:
        for worker_config_file in listdir(WORKERS_CONFIG_DIR):
            filename = Path(worker_config_file).stem
            worker_ids.append(filename)
    elif worker:
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker}.toml"
        if not Path(worker_config_file).is_file():
            message(
                f"The configuration file for worker '{worker}', does not exist in directory '{WORKERS_CONFIG_DIR}'",
                Level.ERROR,
            )
            sys.exit(1)
        filename = Path(worker_config_file).stem
        worker_ids.append(filename)
    else:
        message(
            "Please specify a worker using --worker <worker> or use --all",
            Level.WARNING,
        )
        sys.exit(1)

    return worker_ids


def get_localworker_ids():
    all_worker_ids = get_worker_ids(all_=True)
    local_worker_ids = []
    for worker_id in all_worker_ids:
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        if worker_config["role"] == "LOCALWORKER":
            local_worker_ids.append(worker_id)
    return local_worker_ids


def get_docker_image(c, image, always_pull=False):
    """
    Fetches a docker image locally.

    :param image: The image to pull from dockerhub.
    :param always_pull: Will pull the image even if it exists locally.
    """

    cmd = f"docker images -q {image}"
    _, image_tag = image.split(":")
    result = run(c, cmd, show_ok=False)
    if result.stdout != "" and image_tag != "latest" and image_tag != "dev":
        return

    message(f"Pulling image {image} ...", Level.HEADER)
    cmd = f"docker pull {image}"
    run(c, cmd)
