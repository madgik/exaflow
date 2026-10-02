from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.decomposition.pca_common import (
    PCAReusablePreprocessing,
)
from exaflow.algorithms.exareme3.decomposition.pca_common import run_federated_pca
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput

PCA_COLUMN_CREATOR_NAME = "pca_column_creator"
SUPPORTED_PCA_SCHEMA_VERSION = "1"


class PCAColumnCreator(PreprocessingStep):
    def __init__(self, *, params: Dict[str, object]):
        super().__init__(params=params)
        self._reusable_preprocessing = self._params.get("reusable_preprocessing")
        selected_components = self._params.get("selected_components")
        self._selected_components = (
            list(selected_components)
            if isinstance(selected_components, list)
            else selected_components
        )
        code_prefix = self._params.get("code_prefix", "pca")
        self._code_prefix = str(code_prefix) if code_prefix is not None else ""
        self._recipe: Optional[PCAReusablePreprocessing] = None

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name=PCA_COLUMN_CREATOR_NAME,
            desc="Creates numerical columns from selected PCA components.",
            documentation=(
                "Uses a stored PCA preprocessing recipe to plan selected principal "
                "component columns for downstream analysis. PCA score execution is "
                "introduced in a later checkpoint."
            ),
            label="PCA Column Creator",
            enabled=True,
            parameters={
                "reusable_preprocessing": specs.ParameterSpecification(
                    label="Reusable PCA preprocessing",
                    desc="Stored PCA recipe used to define the source variables and components.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                ),
                "selected_components": specs.ParameterSpecification(
                    label="Selected components",
                    desc="Principal components to create as numerical columns.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
                "code_prefix": specs.ParameterSpecification(
                    label="Code prefix",
                    desc="Prefix used for generated PCA column codes.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="pca",
                ),
            },
            output=specs.PreprocessingOutputSpecification(
                type=specs.PreprocessingOutputType.NEW_NUMERICAL_COLUMN,
                multiple=True,
            ),
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        self._recipe = self._parse_recipe()
        self._validate_selected_components()

        if not self._code_prefix.strip():
            raise BadUserInput("'code_prefix' parameter should not be blank.")

        source_variables = self._recipe.base_parameters.variables
        missing_variables = [
            variable
            for variable in source_variables
            if variable not in inputdata.variables
        ]
        if missing_variables:
            raise BadUserInput(
                "PCA recipe variables are not present in inputdata.variables: "
                f"{missing_variables}."
            )
        missing_metadata = [
            variable for variable in source_variables if variable not in metadata
        ]
        if missing_metadata:
            raise BadUserInput(
                f"PCA recipe variables are missing from metadata: {missing_metadata}."
            )

    def transform_variables(self, *, variables: List[str]) -> List[str]:
        self._validated_recipe()
        return list(variables) + self._output_codes()

    def transform_metadata(self, *, metadata: Dict[str, dict]) -> Dict[str, dict]:
        transformed_metadata = deepcopy(metadata)
        for code in self._output_codes():
            transformed_metadata[code] = {
                "code": code,
                "label": code,
                "sql_type": "real",
                "is_categorical": False,
                "enumerations": None,
            }
        return transformed_metadata

    def transform_data(self, *, data: pd.DataFrame, agg_client) -> pd.DataFrame:
        recipe = self._validated_recipe()
        if recipe.base_parameters.pca_variant == "pca":
            data_transformation = None
        elif recipe.base_parameters.pca_variant == "pca_with_transformation":
            data_transformation = recipe.base_parameters.data_transformation
        else:
            raise BadUserInput(
                f"Unsupported PCA variant: '{recipe.base_parameters.pca_variant}'."
            )

        execution = run_federated_pca(
            agg_client=agg_client,
            data=data,
            variables=recipe.base_parameters.variables,
            data_transformation=data_transformation,
            compute_scores=True,
        )
        scores = execution.scores
        if scores is None or scores.ndim != 2:
            raise BadUserInput("PCA score execution did not return a valid matrix.")
        if scores.shape[0] != len(data):
            raise BadUserInput(
                "PCA score row count does not match the input data row count."
            )
        if scores.shape[1] < len(recipe.component_choices):
            raise BadUserInput(
                "PCA score matrix does not contain all recipe components."
            )

        component_indices = {
            choice.component_id: index
            for index, choice in enumerate(recipe.component_choices)
        }
        for component_id, output_code in zip(
            self._selected_components, self._output_codes()
        ):
            data[output_code] = scores[:, component_indices[component_id]]
        return data

    def _parse_recipe(self) -> PCAReusablePreprocessing:
        try:
            recipe = PCAReusablePreprocessing.model_validate(
                self._reusable_preprocessing
            )
        except (TypeError, ValidationError) as exc:
            raise BadUserInput(
                "'reusable_preprocessing' is not a valid PCA preprocessing recipe."
            ) from exc

        if recipe.preprocessing_name != PCA_COLUMN_CREATOR_NAME:
            raise BadUserInput(
                f"PCA recipe preprocessing_name should be '{PCA_COLUMN_CREATOR_NAME}'."
            )
        if recipe.schema_version != SUPPORTED_PCA_SCHEMA_VERSION:
            raise BadUserInput(
                f"Unsupported PCA recipe schema_version: '{recipe.schema_version}'."
            )

        variant = recipe.base_parameters.pca_variant
        transformation = recipe.base_parameters.data_transformation
        if variant == "pca" and transformation is not None:
            raise BadUserInput(
                "Plain PCA recipes must not include data_transformation."
            )
        if variant == "pca_with_transformation" and transformation is None:
            raise BadUserInput(
                "Transformed PCA recipes must include data_transformation."
            )
        return recipe

    def _validate_selected_components(self) -> None:
        selected_components = self._selected_components
        if not isinstance(selected_components, list) or not selected_components:
            raise BadUserInput(
                "'selected_components' should contain at least one component."
            )
        if not all(isinstance(component, str) for component in selected_components):
            raise BadUserInput("'selected_components' values should be strings.")
        if len(selected_components) != len(set(selected_components)):
            raise BadUserInput("'selected_components' should not contain duplicates.")

        available_components = {
            choice.component_id for choice in self._validated_recipe().component_choices
        }
        unavailable_components = [
            component
            for component in selected_components
            if component not in available_components
        ]
        if unavailable_components:
            raise BadUserInput(
                "Selected PCA components are not available in the recipe: "
                f"{unavailable_components}."
            )

    def _validated_recipe(self) -> PCAReusablePreprocessing:
        if self._recipe is None:
            self._recipe = self._parse_recipe()
        return self._recipe

    def _output_codes(self) -> List[str]:
        return [
            f"{self._code_prefix}_{component}"
            for component in self._selected_components
        ]
