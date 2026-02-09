from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional

from exaflow.algorithms.specifications import AlgorithmName
from exaflow.algorithms.utils.inputdata_utils import Inputdata


class Algorithm(ABC):
    algname: AlgorithmName

    def __init__(
        self,
        *,
        engine,
        metadata: dict,
        inputdata: Inputdata,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._engine = engine
        self._metadata: dict = metadata
        self._inputdata: Inputdata = inputdata
        self._parameters: dict = parameters if parameters is not None else {}

    def __init_subclass__(cls, algname: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def inputdata(self) -> Inputdata:
        return self._inputdata

    def get_parameter(self, name, default=None) -> Any:
        return self._parameters.get(name, default)

    @property
    def drop_na_rows(self) -> bool:
        """
        By default, the rows with 'Not Available' values are dropped.
        If an algorithm needs to keep the 'Not Available' values,
        this method must be overridden to return False.
        """
        return True

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

    def run_local_udf(self, func, kw_args):
        return self._engine.run_udf(
            func,
            self.drop_na_rows,
            self.check_min_rows,
            self.add_dataset_variable,
            kw_args,
        )
