## Test Environments

There are several test areas; the environment-specific suites are summarized below.

### No Environment

The standalone tests are run:

- only with the basic infrastructure docker images pre-built,
- without an environment and
  are supposed to take care (create/cleanup) of any external dependency they need.

### Prod Environment

These tests are run:

- with all (worker, controller, aggregation_server) docker images pre-built,
- with 3 workers, 1 controller and 1 aggregation server,
- with the production algorithms loaded,
- with data paths for each worker configured by the `worker_data_paths_builder.py` script under `tests/test_data/.data_paths/<worker_name>`,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of production nature.

### Algorithm Validation Tests

These tests are run:

- with 1 worker, 1 controller and 1 aggregation server,
- with 5 workers, 1 controller and 1 aggregation server in a different job,
- with data paths for each worker configured by the `worker_data_paths_builder.py` script under `tests/test_data/.data_paths/<worker_name>`,
- with the production algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the workers' information in the `.deployment.toml`.

### Other folders

- `tests/algorithms`, `tests/test_data`, and `tests/testcase_generators` hold demo/test algorithms, test datasets, and test case generators.
