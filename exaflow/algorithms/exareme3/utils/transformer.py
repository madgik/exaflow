import inspect
import json
from abc import ABC
from pathlib import Path

from exaflow.algorithms.specifications import TransformerSpecification


class Transformer(ABC):
    @classmethod
    def get_specification(cls) -> TransformerSpecification:
        transformer_path = Path(inspect.getfile(cls)).resolve()
        transformer_folder = transformer_path.parent

        candidate_paths = [transformer_folder / f"{transformer_path.stem}.json"]
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

        return TransformerSpecification.model_validate(specification)
