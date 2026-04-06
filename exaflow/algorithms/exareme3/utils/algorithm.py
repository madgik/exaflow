import inspect
import json
import math
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.utils.inputdata_utils import Inputdata


class Algorithm(ABC):
    def __init__(
        self,
        *,
        engine,
        inputdata: Inputdata,
        metadata: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._engine = engine
        self._inputdata: Inputdata = inputdata
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self._parameters: dict = parameters if parameters is not None else {}

    @property
    def inputdata(self) -> Inputdata:
        return self._inputdata

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def get_parameter(self, name, default=None) -> Any:
        return self._parameters.get(name, default)

    @property
    def check_min_rows(self) -> bool:
        """
        If an algorithm needs to ignore the minimum row count threshold check,
        for its data, this method must be overridden to return False.
        """
        return True

    @property
    def add_dataset_variable(self) -> bool:
        """
        If an algorithm needs to include the dataset column, in addition to the user
        selected columns, this method should be overridden to return True.
        """
        return False

    @abstractmethod
    def run(self):
        pass

    def run_local_udf(self, func, kw_args, *, identical_results: bool = False):
        results: List[Dict[str, Any]] = self._engine.run_udf(
            func,
            self.check_min_rows,
            self.add_dataset_variable,
            kw_args,
        )
        if not identical_results:
            return results
        if not results:
            raise RuntimeError("UDF returned no results.")

        first = results[0]
        for idx, result in enumerate(results[1:], start=1):
            if not _values_equal(result, first):
                raise RuntimeError(
                    f"Inconsistent UDF responses for '{func.__name__}': "
                    f"worker result at index {idx} differs from index 0."
                )
        return first

    @classmethod
    def get_specification(cls) -> AlgorithmSpecification:
        algorithm_path = Path(inspect.getfile(cls)).resolve()
        algorithm_folder = algorithm_path.parent

        candidate_paths = [algorithm_folder / f"{algorithm_path.stem}.json"]
        specification_path = next(
            (path for path in candidate_paths if path.exists()), None
        )
        if specification_path is None:
            expected = ", ".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(
                f"Specification JSON not found for '{cls.__name__}'. Expected one of: {expected}"
            )

        with specification_path.open("r", encoding="utf-8") as fp:
            specification = json.load(fp)

        return AlgorithmSpecification.model_validate(specification)


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _values_equal(left: Any, right: Any) -> bool:
    if _is_nan(left) and _is_nan(right):
        return True

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal(lv, rv) for lv, rv in zip(left, right))

    return left == right
