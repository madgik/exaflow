import json
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from exaflow import FLOWER_ALGORITHM_FOLDERS
from exaflow import exareme3_algorithm_classes
from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.specifications import PreprocessingStep
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.controller import config as ctrl_config

logger = logging.getLogger(__name__)


def find_spec_paths(folders: str) -> List[Path]:
    paths = (p.strip() for p in folders.split(","))
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
    SpecType = Union[AlgorithmSpecification, PreprocessingStepSpecification]
    enabled_algorithms: Dict[str, AlgorithmSpecification]
    enabled_preprocessing_steps: Dict[str, PreprocessingStepSpecification]

    def __init__(self):
        self._all_specs: Dict[str, Specifications.SpecType] = {}
        self.enabled_algorithms = {}
        self.enabled_preprocessing_steps = {}
        self._flags = {
            ComponentType.AGGREGATION_SERVER: ctrl_config.aggregation_server.enabled,
            ComponentType.FLOWER: ctrl_config.flower.enabled,
        }
        self._load_specifications()
        self._filter_specifications()

    def _load_specifications(self) -> None:
        self._load_exareme3_specifications()
        self._load_flower_specifications()

    def _load_exareme3_specifications(self) -> None:
        self._load_exareme3_algorithm_specifications()
        self._load_exareme3_preprocessing_step_specifications()

    def _load_exareme3_algorithm_specifications(self) -> None:
        for algorithm_name, algorithm_cls in exareme3_algorithm_classes.items():
            spec = algorithm_cls.get_specification()
            if spec.name != algorithm_name:
                raise ValueError(
                    f"Algorithm class key '{algorithm_name}' does not match specification name '{spec.name}'."
                )
            self._add_specification(spec, f"algorithm class '{algorithm_name}'")

    def _load_exareme3_preprocessing_step_specifications(self) -> None:
        for (
            transformer_name,
            transformer_cls,
        ) in exareme3_preprocessing_step_classes.items():
            spec = transformer_cls.get_specification()
            if spec.name != transformer_name:
                raise ValueError(
                    f"Transformer class key '{transformer_name}' does not match specification name '{spec.name}'."
                )
            self._add_specification(spec, f"transformer class '{transformer_name}'")

    def _load_flower_specifications(self) -> None:
        for folder in find_spec_paths(FLOWER_ALGORITHM_FOLDERS):
            for file in folder.glob("*.json"):
                raw = load_json_file(file)
                spec_cls = self._choose_spec_class(raw)
                if spec_cls is not AlgorithmSpecification:
                    continue
                spec = AlgorithmSpecification.model_validate(raw)
                self._add_specification(spec, str(file))

    def _add_specification(self, spec: SpecType, source: str) -> None:
        if spec.name in self._all_specs:
            raise ValueError(f"Duplicate spec '{spec.name}' in {source}")
        self._all_specs[spec.name] = spec

    def _filter_specifications(self) -> None:
        for spec in self._all_specs.values():
            if not spec.enabled:
                continue

            if spec.components and (
                not any(self._flags.get(dt, False) for dt in spec.components)
            ):
                missing = spec.components[0].value if spec.components else "unspecified"
                logger.warning(
                    f"{type(spec).__name__} '{spec.name}' skipped: '{missing}' not deployed"
                )
                continue

            if isinstance(spec, AlgorithmSpecification):
                self.enabled_algorithms[spec.name] = spec
            else:
                self.enabled_preprocessing_steps[spec.name] = spec

    @staticmethod
    def _choose_spec_class(raw: dict) -> Type[SpecType]:
        if raw["type"].startswith(PreprocessingStep.EXAREME3_PREPROCESSING_STEP.value):
            return PreprocessingStepSpecification
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
