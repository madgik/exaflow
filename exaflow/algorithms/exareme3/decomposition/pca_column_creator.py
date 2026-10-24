from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.decomposition.pca import PCAReusablePreprocessing
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput

PCA_COLUMN_CREATOR_NAME = "pca_column_creator"
SUPPORTED_PCA_SCHEMA_VERSION = "1"


class PCAColumnCreator(PreprocessingStep):
    def __init__(self, *, params: Dict[str, object]):
        super().__init__(params=params)
        self._reusable_preprocessing = self._params.get("reusable_preprocessing")
        source_variables = self._params.get("variables")
        self._source_variables = (
            list(source_variables)
            if isinstance(source_variables, list)
            else source_variables
        )
        selected_components = self._params.get("selected_components")
        self._selected_components = (
            list(selected_components)
            if isinstance(selected_components, list)
            else selected_components
        )
        new_column_names = self._params.get("new_column_names")
        self._new_column_names = (
            list(new_column_names)
            if isinstance(new_column_names, list)
            else new_column_names
        )
        self._n_components = self._params.get("n_components")
        self._recipe: Optional[PCAReusablePreprocessing] = None

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name=PCA_COLUMN_CREATOR_NAME,
            desc="Creates numerical columns from selected PCA components.",
            documentation=(
                "Creates selected principal component score columns either from a "
                "stored PCA recipe or by fitting PCA directly on explicitly selected "
                "numerical variables."
            ),
            label="PCA Column Creator",
            enabled=True,
            parameters={
                "reusable_preprocessing": specs.ParameterSpecification(
                    label="Reusable PCA preprocessing",
                    desc="Optional PCA recipe used to define the source variables and components.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                ),
                "variables": specs.ParameterSpecification(
                    label="PCA variables",
                    desc="Numerical variables used when no PCA recipe is provided.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=True,
                ),
                "selected_components": specs.ParameterSpecification(
                    label="Selected components",
                    desc="Principal components to create as numerical columns.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
                "new_column_names": specs.ParameterSpecification(
                    label="New column names",
                    desc="Names assigned to the selected PCA score columns.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
                "n_components": specs.ParameterSpecification(
                    label="Number of components",
                    desc="Number of PCA score columns to create.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    min=1,
                ),
            },
            outputs=specs.PreprocessingOutputsSpecification(
                new_column_names_param="new_column_names",
                columns=[
                    specs.PreprocessingOutputColumnSpecification(
                        type=specs.InputDataType.REAL,
                        stattype=specs.InputDataStatType.NUMERICAL,
                        column_count_param="n_components",
                    )
                ],
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
        if self._reusable_preprocessing is not None:
            self._recipe = self._parse_recipe()
            source_variables = self._recipe.base_parameters.variables
        else:
            self._validate_source_variables()
            source_variables = self._source_variables
        self._validate_selected_components()
        if self._n_components != len(self._selected_components):
            raise BadUserInput(
                "'n_components' must match the number of selected PCA components."
            )

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
        return list(variables) + list(self._new_column_names)

    def transform_metadata(self, *, metadata: Dict[str, dict]) -> Dict[str, dict]:
        transformed_metadata = deepcopy(metadata)
        transformed_metadata.update(
            resolve_output_metadata(self.get_specification(), self._params, metadata)
        )
        return transformed_metadata

    def transform_data(self, *, data: pd.DataFrame, agg_client) -> pd.DataFrame:
        recipe = self._validated_recipe()
        source_variables = (
            recipe.base_parameters.variables
            if recipe is not None
            else self._source_variables
        )
        X_values = np.array(data.loc[:, source_variables], dtype=float, copy=True)
        model = FederatedPCA(agg_client=agg_client, copy=False)
        scores = model.fit_transform(X_values)
        if scores is None or scores.ndim != 2:
            raise BadUserInput("PCA score execution did not return a valid matrix.")
        if scores.shape[0] != len(data):
            raise BadUserInput(
                "PCA score row count does not match the input data row count."
            )
        if recipe is not None:
            component_indices = {
                choice.component_id: index
                for index, choice in enumerate(recipe.component_choices)
            }
        else:
            component_indices = {
                f"PC{index + 1}": index for index in range(scores.shape[1])
            }
        for component_id, output_code in zip(
            self._selected_components, self._new_column_names
        ):
            if component_id not in component_indices:
                raise BadUserInput(
                    f"Selected PCA component is not available: {component_id}."
                )
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

        return recipe

    def _validate_selected_components(self) -> None:
        selected_components = self._selected_components
        if len(selected_components) != len(set(selected_components)):
            raise BadUserInput("'selected_components' should not contain duplicates.")

        recipe = self._validated_recipe()
        if recipe is None:
            invalid_components = [
                component
                for component in selected_components
                if not component.startswith("PC")
                or not component[2:].isdigit()
                or int(component[2:]) < 1
            ]
            if invalid_components:
                raise BadUserInput(
                    "Selected PCA components should use positive PC numbers: "
                    f"{invalid_components}."
                )
            return

        available_components = {
            choice.component_id for choice in recipe.component_choices
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

    def _validated_recipe(self) -> Optional[PCAReusablePreprocessing]:
        if self._recipe is None:
            if self._reusable_preprocessing is None:
                return None
            self._recipe = self._parse_recipe()
        return self._recipe

    def _validate_source_variables(self) -> None:
        if self._source_variables is None:
            raise BadUserInput(
                "'variables' should contain at least one variable when "
                "'reusable_preprocessing' is not provided."
            )
        if not all(variable.strip() for variable in self._source_variables):
            raise BadUserInput("'variables' values should be nonblank strings.")
        if len(self._source_variables) != len(set(self._source_variables)):
            raise BadUserInput("'variables' should not contain duplicates.")
