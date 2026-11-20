"""Derive standard-scaled columns without replacing their sources."""

from copy import deepcopy

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.feature_engineering import FederatedStandardScaler
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput


class StandardScalerStep(PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return specs.PreprocessingStepSpecification(
            name="standard_scaler",
            label="Standard Scaler",
            desc="Create standardized numerical columns while preserving the sources.",
            documentation=(
                "Subtract the pooled mean and divide by population standard "
                "deviation (ddof=0), computed from aggregated sufficient statistics "
                "without sharing raw data. Select distinct numerical source_variables "
                "and supply equally many new_column_names in matching order. "
                "Each source is scaled independently. Constant or "
                "numerically indistinguishable scales use one. Missing values "
                "are ignored during fitting and remain missing in the output; "
                "a source with no observed values is rejected. Infinite values "
                "are rejected. Empty participants contribute neutral statistics. "
                "The original columns are retained within the request. "
                "The fitted object is internal to the execution; no recipe or "
                "cross-request replay parameter is provided. Methodology is "
                "aligned with scikit-learn StandardScaler."
            ),
            enabled=True,
            parameters={
                "source_variables": specs.ParameterSpecification(
                    label="Source variables",
                    desc="Numerical variables to standardize.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["variables"],
                    ),
                ),
                "new_column_names": specs.ParameterSpecification(
                    label="New column names",
                    desc="Names assigned to the standardized outputs in source order.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
            },
            outputs=specs.PreprocessingOutputsSpecification(
                new_column_names_param="new_column_names",
                columns=[
                    specs.PreprocessingOutputColumnSpecification(
                        type=specs.InputDataType.REAL,
                        stattype=specs.InputDataStatType.NUMERICAL,
                        column_count_param="source_variables",
                    )
                ],
            ),
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def validate_params(self, *, inputdata, metadata):
        sources = self._params["source_variables"]
        if len(set(sources)) != len(sources):
            raise BadUserInput("Select distinct source variables.")
        for source in sources:
            source_metadata = metadata[source]
            if source_metadata.get("is_categorical") or source_metadata.get(
                "sql_type"
            ) not in {"int", "real"}:
                raise BadUserInput("Select numerical source variables.")

    def transform_variables(self, *, variables):
        return [*variables, *self._params["new_column_names"]]

    def transform_metadata(self, *, metadata):
        outputs = resolve_output_metadata(
            self.get_specification(), self._params, metadata
        )
        return {**deepcopy(metadata), **outputs}

    def transform_data(self, *, data, agg_client=None):
        sources = self._params["source_variables"]
        names = self._params["new_column_names"]
        scaler = FederatedStandardScaler()
        scaler.fit(
            agg_client=agg_client,
            data=data,
            categorical_vars=[],
            numerical_vars=sources,
        )
        data[names] = scaler.transform(
            data, categorical_vars=[], numerical_vars=sources
        )
        return data
