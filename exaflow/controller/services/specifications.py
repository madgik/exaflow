import json
import logging
import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from exaflow import EXAREME3_ALGORITHM_FOLDERS
from exaflow import FLOWER_ALGORITHM_FOLDERS
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.specifications import TransformerSpecification
from exaflow.algorithms.specifications import TransformerType
from exaflow.controller import config as ctrl_config

logger = logging.getLogger(__name__)


def find_spec_paths() -> List[Path]:
    folders = (
        FLOWER_ALGORITHM_FOLDERS,
        EXAREME3_ALGORITHM_FOLDERS,
    )
    paths = (p.strip() for folder in folders for p in folder.split(","))
    return [Path(p) for p in paths if p]


def load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as err:
        logger.error("Invalid JSON in %s: %s", path, err)
        raise
    except OSError as err:
        logger.error("Failed to read %s: %s", path, err)
        raise


class Specifications:
    SpecType = Union[AlgorithmSpecification, TransformerSpecification]
    enabled_algorithms: Dict[str, AlgorithmSpecification]
    enabled_transformers: Dict[str, TransformerSpecification]

    def __init__(self):
        self._all_specs: Dict[str, Specifications.SpecType] = {}
        self.enabled_algorithms = {}
        self.enabled_transformers = {}
        self._flags = {
            ComponentType.AGGREGATION_SERVER: ctrl_config.aggregation_server.enabled,
            ComponentType.FLOWER: ctrl_config.flower.enabled,
        }
        self._load_specifications()
        self._filter_specifications()

    def _load_specifications(self) -> None:
        for folder in find_spec_paths():
            for file in folder.glob("*.json"):
                raw = load_json_file(file)
                spec_cls = self._choose_spec_class(raw)
                spec = spec_cls.parse_obj(raw)
                name = spec.name
                if name in self._all_specs:
                    raise ValueError(f"Duplicate spec '{name}' in {file}")
                self._all_specs[name] = spec

    def _filter_specifications(self) -> None:
        for spec in self._all_specs.values():
            if not spec.enabled:
                continue

            if spec.components and (
                not any(self._flags.get(dt, False) for dt in spec.components)
            ):
                missing = spec.components[0].value if spec.components else "unspecified"
                warnings.warn(
                    f"{type(spec).__name__} '{spec.name}' skipped: '{missing}' not deployed"
                )
                continue

            if isinstance(spec, AlgorithmSpecification):
                self.enabled_algorithms[spec.name] = spec
            else:
                self.enabled_transformers[spec.name] = spec

    @staticmethod
    def _choose_spec_class(raw: dict) -> Type[SpecType]:
        if raw["type"].startswith(TransformerType.EXAREME3_TRANSFORMER.value):
            return TransformerSpecification
        return AlgorithmSpecification

    def get_algorithm_type(self, algo_name: str) -> AlgorithmType:
        if algo_name not in self.enabled_algorithms:
            raise KeyError(f"Algorithm '{algo_name}' not enabled or not found.")
        return self.enabled_algorithms[algo_name].type

    def get_component_types(self, algo_name: str) -> List[ComponentType]:
        if algo_name not in self.enabled_algorithms:
            raise KeyError(f"Algorithm '{algo_name}' not enabled or not found.")
        return self.enabled_algorithms[algo_name].components


specifications = Specifications()
