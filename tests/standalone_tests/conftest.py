import pytest

from exaflow.controller.services.controller_shutdown import reset_controller_shutdown


@pytest.fixture(autouse=True)
def _reset_controller_shutdown():
    """The shutdown flag is process-wide: it must not leak into the next test.

    Once it is set, every RPC registered afterwards is cancelled at once.
    """
    yield
    reset_controller_shutdown()
