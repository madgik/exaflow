import asyncio
import types

import pytest

from exaflow.controller.services.errors import GlobalWorkerUnavailableError
from exaflow.controller.services.flower.strategies import FlowerStrategy
from tests.standalone_tests.controller.analysis_support import build_strategy


def test_flower_analysis_without_global_worker_reports_it_unavailable():
    strategy = build_strategy(FlowerStrategy)
    strategy._global_worker_tasks_handler = None
    strategy._local_worker_tasks_handlers = [
        types.SimpleNamespace(worker_id="localworker1")
    ]
    strategy._controller.algorithm_execution_lock = asyncio.Lock()

    with pytest.raises(GlobalWorkerUnavailableError):
        asyncio.run(strategy.execute())
