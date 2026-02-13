# FederationTestTemplate Documentation

The `FederationTestTemplate` is an abstract base class designed to facilitate the testing and verification of federated algorithms by comparing them against a centralized (global) execution.

## Key Features

- **Automated Process Management**: Handles the spawning and termination of the gRPC server and multiple client processes.
- **Data Partitioning**: Automatically handles dataset splitting between federated clients.
- **Consistency Verification**: Forces implementation of both federated and centralized logic to ensure results match.

## Core Methods

### Abstract Methods (To Implement)

Subclasses must implement the following methods:

- `federated_computation(self, local_dataset)`: Logic to be executed by each federated client.
- `centralized_computation(self, global_dataset)`: Logic to be executed on the full dataset for ground truth.
- `compare(self, federated_output, global_output)`: Logic to assert or print comparison results.

### Class Methods (To Run)

- `run_test(dataset, client_count=3)`: The primary entry point to start a full test suite.
    - Spawns a server process.
    - Spawns `client_count` client processes.
    - Each client instantiates the test class and executes the computation.
    - Automatically cleans up processes after completion.

## Example Usage

To create a new test, inherit from `FederationTestTemplate` and implement the abstract methods.

```python
from tests_new.utils.test_template import FederationTestTemplate


class MyTest(FederationTestTemplate):
  def federated_computation(self, local_dataset):
    # Your federated logic here
    return local_dataset.mean()

  def centralized_computation(self, centralized_dataset):
    # Your centralized logic here
    return centralized_dataset.mean()

  def compare(self, federated_output, global_output):
    print(f"Fed: {federated_output}, Global: {global_output}")
    assert federated_output == global_output


# To run the test:
if __name__ == "__main__":
  from tests_new.utils.datasets.dummy_dataset import DummyDataset

  MyTest.run_test(DummyDataset(), client_count=3)
```

## Process management

- `run_test(dataset, client_count=3)`: The primary entry point to start a full test suite.
    - Spawns a server process using a top-level helper.
    - Spawns `client_count` client processes by directly instantiating the calling class.
    - Automatically cleans up processes after completion.
