from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


@unique
class AlgorithmType(Enum):
    FLOWER = "flower"
    EXAREME3 = "exareme3"


@unique
class PreprocessingStepType(Enum):
    EXAREME3_PREPROCESSING_STEP = "exareme3_preprocessing_step"


@unique
class InputDataType(Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    JSONOBJECT = "jsonObject"


@unique
class InputDataStatType(Enum):
    NUMERICAL = "numerical"
    NOMINAL = "nominal"


@unique
class ParameterType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    BOOLEAN = "boolean"
    DICT = "dict"


@unique
class ParameterDictValueType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    BOOLEAN = "boolean"
    FILTER = "filter"


@unique
class PreprocessingOutputType(str, Enum):
    NEW_CATEGORICAL_COLUMN = "new_categorical_column"


@unique
class ParameterEnumType(str, Enum):
    LIST = "list"
    INPUT_VAR_CDE_ENUMS = "input_var_CDE_enums"
    FIXED_VAR_CDE_ENUMS = "fixed_var_CDE_enums"
    INPUT_VAR_NAMES = "input_var_names"


@unique
class PreprocessingStepName(str, Enum):
    CATEGORICAL_COLUMN_CREATOR = "categorical_column_creator"
    LONGITUDINAL_TRANSFORMER = "longitudinal_transformer"
    MISSING_VALUES_HANDLER = "missing_values_handler"
    OUTLIER_WINSORIZER = "outlier_winsorizer"

    def __str__(self) -> str:
        return str.__str__(self)


@unique
class AlgorithmName(str, Enum):
    ANOVA_ONEWAY = "anova_oneway"
    ANOVA_TWOWAY = "anova_twoway"
    COX_REGRESSION_CLASSICAL = "cox_regression_classical"
    DESCRIBE = "describe"
    KMEANS = "kmeans"
    LINEAR_REGRESSION = "linear_regression"
    LINEAR_REGRESSION_CV = "linear_regression_cv"
    LOGISTIC_REGRESSION = "logistic_regression"
    LOGISTIC_REGRESSION_CV = "logistic_regression_cv"
    LMM = "lmm"
    GLMM_BINARY = "glmm_binary"
    GLMM_ORDINAL = "glmm_ordinal"
    HISTOGRAM = "histogram"
    NAIVE_BAYES_GAUSSIAN = "naive_bayes_gaussian"
    NAIVE_BAYES_CATEGORICAL = "naive_bayes_categorical"
    NAIVE_BAYES_CATEGORICAL_CV = "naive_bayes_categorical_cv"
    NAIVE_BAYES_GAUSSIAN_CV = "naive_bayes_gaussian_cv"
    PCA = "pca"
    PCA_WITH_TRANSFORMATION = "pca_with_transformation"
    PEARSON_CORRELATION = "pearson_correlation"
    LINEAR_SVM = "linear_svm"
    TTEST_INDEPENDENT = "ttest_independent"
    TTEST_ONESAMPLE = "ttest_onesample"
    TTEST_PAIRED = "ttest_paired"
    COX_REGRESSION_STACKED = "cox_regression_stacked"
    OUTLIER_REPORT = "outlier_report"

    def __str__(self) -> str:
        return str.__str__(self)


@unique
class ComponentType(str, Enum):
    AGGREGATION_SERVER = "AGGREGATION_SERVER"
    FLOWER = "FLOWER"


class ImmutableBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class InputDataSpecification(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    stattypes: List[InputDataStatType]
    required: bool
    min_count: Optional[int] = None
    max_count: Optional[int] = None

    @model_validator(mode="after")
    def validate_count_bounds(self):
        if self.min_count is not None and self.min_count < 0:
            raise ValueError(
                f"In inputdata '{self.label}', 'min_count' should be greater than or equal to 0."
            )
        if self.max_count is not None and self.max_count < 0:
            raise ValueError(
                f"In inputdata '{self.label}', 'max_count' should be greater than or equal to 0."
            )
        if (
            self.min_count is not None
            and self.max_count is not None
            and self.min_count > self.max_count
        ):
            raise ValueError(
                f"In inputdata '{self.label}', 'min_count' cannot be greater than 'max_count'."
            )
        return self


class ParameterEnumSpecification(ImmutableBaseModel):
    type: ParameterEnumType
    source: List[str]


class ParameterSpecification(ImmutableBaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    required: bool
    multiple: bool
    default: Any = None
    enums: Optional[ParameterEnumSpecification] = None
    dict_keys_enums: Optional[ParameterEnumSpecification] = None
    dict_values_type: Optional[ParameterDictValueType] = None
    dict_values_enums: Optional[ParameterEnumSpecification] = None
    min: Optional[float] = None
    max: Optional[float] = None


class PreprocessingOutputSpecification(ImmutableBaseModel):
    type: PreprocessingOutputType
    code_parameter: Optional[str] = None


class WorkflowStepSpecification(ImmutableBaseModel):
    name: str
    desc: str
    documentation: str
    label: str
    enabled: bool
    parameters: Optional[Dict[str, ParameterSpecification]] = None
    components: List[ComponentType] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_parameters(self):
        if not self.parameters:
            return self

        cls_values = {
            "label": self.label,
            "x": getattr(self, "x", None),
            "y": getattr(self, "y", None),
        }
        for param_value in self.parameters.values():
            _validate_parameter_enums(param_value, cls_values)
            _validate_parameter_type_dict(param_value, cls_values)
            _validate_parameter_type_dict_enums(param_value, cls_values)
        return self


class AlgorithmSpecification(WorkflowStepSpecification):
    y: InputDataSpecification
    x: Optional[InputDataSpecification] = None
    requires_validation_datasets: bool = False
    type: AlgorithmType = AlgorithmType.EXAREME3
    required_preprocessing: List[str] = Field(default_factory=list)


class PreprocessingStepSpecification(WorkflowStepSpecification):
    output: Optional[PreprocessingOutputSpecification] = None
    type: PreprocessingStepType = PreprocessingStepType.EXAREME3_PREPROCESSING_STEP


def _validate_parameter_with_enums_type_fixed_var_CDE_enums(param_value, cls_values):
    if len(param_value.enums.source) != 1:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'fixed_var_CDE_enums' "
            f"that supports only one value. Value given: {param_value.enums.source}."
        )


def _validate_parameter_with_enums_type_input_var_CDE_enums(param_value, cls_values):
    if len(param_value.enums.source) != 1:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'input_var_CDE_enums' "
            f"that supports only one value. Value given: {param_value.enums.source}."
        )

    value = param_value.enums.source[0]  # Only one value is allowed
    if value not in ["x", "y"]:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'input_var_CDE_enums' "
            f"that supports only 'x' or 'y' as source. Value given: '{value}'."
        )
    if param_value.multiple:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'input_var_CDE_enums' "
            f"that doesn't support 'multiple=True', in the parameter."
        )
    variable_spec = cls_values["x"] if value == "x" else cls_values["y"]
    if variable_spec is None:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_CDE_ENUMS.value}' with source '{value}', but the algorithm "
            f"does not define that variable specification."
        )
    if variable_spec.max_count is None or variable_spec.max_count > 1:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_CDE_ENUMS.value}' "
            f"that requires max_count=1 in its linked variable specification '{variable_spec.label}'."
        )


def _validate_parameter_with_enums_type_input_var_names(param_value, cls_values):
    if param_value.types != [ParameterType.TEXT]:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_NAMES.value}' that supports ONLY 'types=[\"text\"]' but the 'types' "
            f"provided were {[t.value for t in param_value.types]}."
        )
    _validate_input_var_names_enum_sources(
        parameter_label=param_value.label,
        enum_spec=param_value.enums,
        cls_values=cls_values,
    )


def _validate_input_var_names_enum_sources(
    *,
    parameter_label: str,
    enum_spec: ParameterEnumSpecification,
    cls_values,
):
    has_algorithm_input_specs = (
        cls_values.get("x") is not None or cls_values.get("y") is not None
    )
    allowed_sources = {"x", "y"} if has_algorithm_input_specs else {"variables"}
    invalid_sources = [
        source for source in enum_spec.source if source not in allowed_sources
    ]
    if invalid_sources:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{parameter_label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_NAMES.value}' with unsupported sources {invalid_sources}. "
            f"Allowed sources are {sorted(allowed_sources)}."
        )


def _validate_parameter_enums(param_value, cls_values):
    if not param_value.enums:
        return
    if param_value.enums.type == ParameterEnumType.FIXED_VAR_CDE_ENUMS:
        _validate_parameter_with_enums_type_fixed_var_CDE_enums(param_value, cls_values)
    elif param_value.enums.type == ParameterEnumType.INPUT_VAR_CDE_ENUMS:
        _validate_parameter_with_enums_type_input_var_CDE_enums(param_value, cls_values)
    elif param_value.enums.type == ParameterEnumType.INPUT_VAR_NAMES:
        _validate_parameter_with_enums_type_input_var_names(param_value, cls_values)


def _validate_parameter_type_dict(param_value, cls_values):
    if ParameterType.DICT in param_value.types:
        if len(param_value.types) > 1:
            raise ValueError(
                f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' cannot use 'dict' type "
                f"combined with other types. Types provided: {param_value.types}. "
            )


def _validate_parameter_type_dict_keys_enums(param_value, cls_values):
    if not param_value.dict_keys_enums:
        return

    if ParameterType.DICT not in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'dict_keys_enums' "
            f"but the allowed 'types' is not '{ParameterType.DICT}'."
        )
    if param_value.dict_keys_enums.type == ParameterEnumType.INPUT_VAR_NAMES:
        _validate_input_var_names_enum_sources(
            parameter_label=param_value.label,
            enum_spec=param_value.dict_keys_enums,
            cls_values=cls_values,
        )


def _validate_parameter_type_dict_values_enums(param_value, cls_values):
    if not param_value.dict_values_enums:
        return

    if ParameterType.DICT not in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'dict_values_enums' "
            f"but the allowed 'types' is not '{ParameterType.DICT}'."
        )


def _validate_parameter_type_dict_values_type(param_value, cls_values):
    if not param_value.dict_values_type:
        return

    if ParameterType.DICT not in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'dict_values_type' "
            f"but the allowed 'types' is not '{ParameterType.DICT}'."
        )


def _validate_parameter_type_dict_enums_not_allowed(param_value, cls_values):
    if not param_value.enums:
        return

    if ParameterType.DICT in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'enums' "
            f"but since the 'types' is '{ParameterType.DICT}', you should use 'dict_keys_enums' and 'dict_values_enums'."
        )


def _validate_parameter_type_dict_enums(param_value, cls_values):
    _validate_parameter_type_dict_keys_enums(param_value, cls_values)
    _validate_parameter_type_dict_values_type(param_value, cls_values)
    _validate_parameter_type_dict_values_enums(param_value, cls_values)
    _validate_parameter_type_dict_enums_not_allowed(param_value, cls_values)
