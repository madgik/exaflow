"""RunConfig validation and env mapping for flowertune_llm_medical."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import root_validator
from pydantic import validator


class StrictBaseModel(BaseModel):
    """Strict model base used for run configuration validation."""

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        validate_assignment = True


class Runtime(str, Enum):
    SIMULATION = "simulation"


class Quantization(str, Enum):
    NONE = "none"
    BIT_8 = "8bit"
    BIT_4 = "4bit"


class PeftMethod(str, Enum):
    LORA = "lora"


class Partitioner(str, Enum):
    IID = "iid"


class PartitionIdStrategy(str, Enum):
    FLOWER_SIMULATION = "flower_simulation"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class TaskType(str, Enum):
    CAUSAL_LM = "causal_lm"


class ModelBackend(str, Enum):
    TINY = "tiny"
    HF_PEFT = "hf_peft"


class EvalMetric(str, Enum):
    LOSS = "loss"
    PERPLEXITY = "perplexity"


class FederationConfig(StrictBaseModel):
    num_rounds: int = Field(..., ge=1)
    num_clients: int = Field(..., ge=1)
    fraction_fit: float = Field(1.0, gt=0.0, le=1.0)
    fraction_evaluate: float = Field(1.0, ge=0.0, le=1.0)
    min_fit_clients: int = Field(..., ge=1)
    min_evaluate_clients: int = Field(..., ge=0)
    min_available_clients: int = Field(..., ge=1)

    @root_validator(skip_on_failure=True)
    def _validate_client_counts(cls, values):
        total_clients = values["num_clients"]
        for key in ("min_fit_clients", "min_evaluate_clients", "min_available_clients"):
            if values[key] > total_clients:
                raise ValueError(f"{key} must be <= num_clients")
        return values


class ModelConfig(StrictBaseModel):
    backend: ModelBackend = ModelBackend.TINY
    model_name: str = Field(..., min_length=1)
    task_type: TaskType = TaskType.CAUSAL_LM
    max_seq_length: int = Field(512, ge=16)
    quantization: Quantization = Quantization.BIT_4


class PeftConfig(StrictBaseModel):
    enabled: bool
    method: Optional[PeftMethod] = None
    r: Optional[int] = Field(None, ge=1)
    alpha: Optional[int] = Field(None, ge=1)
    dropout: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_modules: Optional[List[str]] = None

    @root_validator(skip_on_failure=True)
    def _validate_enabled_fields(cls, values):
        if not values["enabled"]:
            return values
        required = ("method", "r", "alpha")
        missing = [name for name in required if values.get(name) is None]
        if missing:
            raise ValueError(
                f"peft enabled=true requires these fields: {', '.join(missing)}"
            )
        return values

    @validator("target_modules")
    def _validate_target_modules(cls, value):
        if value is None:
            return value
        if not value:
            raise ValueError("peft.target_modules must not be empty when provided")
        return value


class OptimizerConfig(StrictBaseModel):
    learning_rate: float = Field(2e-4, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)


class LocalTrainingConfig(StrictBaseModel):
    batch_size: int = Field(..., ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    local_steps: Optional[int] = Field(None, ge=1)
    epochs: Optional[float] = Field(None, gt=0.0)
    fp16: bool = False
    bf16: bool = True

    @root_validator(skip_on_failure=True)
    def _validate_steps_xor_epochs(cls, values):
        has_steps = values.get("local_steps") is not None
        has_epochs = values.get("epochs") is not None
        if has_steps == has_epochs:
            raise ValueError("exactly one of local_steps or epochs must be set")
        if values.get("fp16") and values.get("bf16"):
            raise ValueError("fp16 and bf16 cannot both be true")
        return values


class DatasetConfig(StrictBaseModel):
    dataset_name: str = Field(..., min_length=1)
    dataset_config: Optional[str] = None
    split: str = "train"
    partitioner: Partitioner = Partitioner.IID
    num_partitions: int = Field(..., ge=1)
    partition_id_strategy: PartitionIdStrategy = PartitionIdStrategy.FLOWER_SIMULATION
    val_split_ratio: float = Field(0.1, ge=0.0, lt=1.0)


class EvaluationConfig(StrictBaseModel):
    enabled: bool = True
    evaluate_every_n_rounds: int = Field(1, ge=1)
    metrics: List[EvalMetric] = Field(
        default_factory=lambda: [EvalMetric.LOSS, EvalMetric.PERPLEXITY]
    )


class ArtifactsConfig(StrictBaseModel):
    artifact_dir: str = Field(..., min_length=1)
    checkpoint_every_n_rounds: int = Field(1, ge=1)
    save_final_model: bool = True
    save_optimizer_state: bool = False


class LoggingConfig(StrictBaseModel):
    log_level: LogLevel = LogLevel.INFO
    report_round_metrics: bool = True
    report_client_metrics: bool = False


class TimeoutsConfig(StrictBaseModel):
    round_timeout_sec: int = Field(1800, ge=1)
    job_timeout_sec: int = Field(14400, ge=1)


class RunConfigContract(StrictBaseModel):
    schema_version: str = "1.1"
    runtime: Runtime = Runtime.SIMULATION
    seed: int = Field(42, ge=0)
    federation: FederationConfig
    model: ModelConfig
    peft: PeftConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    local_training: LocalTrainingConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)

    @validator("schema_version")
    def _validate_schema_version(cls, value):
        if value != "1.1":
            raise ValueError("schema_version must be '1.1'")
        return value

    @root_validator(skip_on_failure=True)
    def _validate_cross_section_constraints(cls, values):
        federation = values["federation"]
        dataset = values["dataset"]
        evaluation = values["evaluation"]
        timeouts = values["timeouts"]
        artifacts = values["artifacts"]

        if dataset.partitioner == Partitioner.IID and (
            dataset.num_partitions != federation.num_clients
        ):
            raise ValueError(
                "for iid partitioner, dataset.num_partitions must equal federation.num_clients"
            )
        if not evaluation.enabled:
            if federation.fraction_evaluate != 0:
                raise ValueError(
                    "federation.fraction_evaluate must be 0 when evaluation.enabled=false"
                )
            if federation.min_evaluate_clients != 0:
                raise ValueError(
                    "federation.min_evaluate_clients must be 0 when evaluation.enabled=false"
                )
        if timeouts.job_timeout_sec < timeouts.round_timeout_sec:
            raise ValueError("timeouts.job_timeout_sec must be >= round_timeout_sec")
        if artifacts.checkpoint_every_n_rounds > federation.num_rounds:
            raise ValueError(
                "artifacts.checkpoint_every_n_rounds must be <= federation.num_rounds"
            )
        return values


def parse_run_config(parameters: Dict[str, Any]) -> RunConfigContract:
    """Parse and validate raw parameters into a strict run config model."""

    return RunConfigContract.parse_obj(parameters or {})


def serialize_run_config(config: RunConfigContract) -> Dict[str, Any]:
    """Return canonical, serializable run config dictionary."""

    return config.dict(exclude_none=True)


def build_env_mapping(config: RunConfigContract, request_id: str) -> Dict[str, str]:
    """Build canonical env var mapping from a validated run configuration."""

    def _bool(v: bool) -> str:
        return "true" if v else "false"

    def _set(
        env: Dict[str, str],
        key: str,
        value: Any,
        *,
        as_bool: bool = False,
        as_json: bool = False,
    ) -> None:
        if value is None:
            return
        if as_bool:
            env[key] = _bool(bool(value))
            return
        if as_json:
            env[key] = json.dumps(value)
            return
        env[key] = str(value)

    resolved_artifact_dir = config.artifacts.artifact_dir.replace(
        "${request_id}", request_id
    )
    env: Dict[str, str] = {}
    _set(env, "RUN_CONFIG_SCHEMA_VERSION", config.schema_version)
    _set(env, "RUN_RUNTIME", config.runtime.value)
    _set(env, "SEED", config.seed)

    _set(env, "NUM_ROUNDS", config.federation.num_rounds)
    _set(env, "NUM_CLIENTS", config.federation.num_clients)
    _set(env, "FRACTION_FIT", config.federation.fraction_fit)
    _set(env, "FRACTION_EVALUATE", config.federation.fraction_evaluate)
    _set(env, "MIN_FIT_CLIENTS", config.federation.min_fit_clients)
    _set(env, "MIN_EVALUATE_CLIENTS", config.federation.min_evaluate_clients)
    _set(env, "MIN_AVAILABLE_CLIENTS", config.federation.min_available_clients)

    _set(env, "MODEL_NAME", config.model.model_name)
    _set(env, "MODEL_BACKEND", config.model.backend.value)
    _set(env, "TASK_TYPE", config.model.task_type.value)
    _set(env, "MAX_SEQ_LENGTH", config.model.max_seq_length)
    _set(env, "QUANTIZATION", config.model.quantization.value)

    _set(env, "PEFT_ENABLED", config.peft.enabled, as_bool=True)
    _set(env, "PEFT_METHOD", config.peft.method.value if config.peft.method else None)
    _set(env, "LORA_R", config.peft.r)
    _set(env, "LORA_ALPHA", config.peft.alpha)
    _set(env, "LORA_DROPOUT", config.peft.dropout)
    _set(env, "LORA_TARGET_MODULES", config.peft.target_modules, as_json=True)

    _set(env, "LEARNING_RATE", config.optimizer.learning_rate)
    _set(env, "WEIGHT_DECAY", config.optimizer.weight_decay)
    _set(env, "MAX_GRAD_NORM", config.optimizer.max_grad_norm)

    _set(env, "BATCH_SIZE", config.local_training.batch_size)
    _set(env, "GRAD_ACC_STEPS", config.local_training.gradient_accumulation_steps)
    _set(env, "LOCAL_STEPS", config.local_training.local_steps)
    _set(env, "LOCAL_EPOCHS", config.local_training.epochs)
    _set(env, "FP16", config.local_training.fp16, as_bool=True)
    _set(env, "BF16", config.local_training.bf16, as_bool=True)

    _set(env, "DATASET_NAME", config.dataset.dataset_name)
    _set(env, "DATASET_CONFIG", config.dataset.dataset_config)
    _set(env, "DATASET_SPLIT", config.dataset.split)
    _set(env, "PARTITIONER", config.dataset.partitioner.value)
    _set(env, "NUM_PARTITIONS", config.dataset.num_partitions)
    _set(env, "PARTITION_ID_STRATEGY", config.dataset.partition_id_strategy.value)
    _set(env, "VAL_SPLIT_RATIO", config.dataset.val_split_ratio)

    _set(env, "EVAL_ENABLED", config.evaluation.enabled, as_bool=True)
    _set(env, "EVAL_EVERY_N_ROUNDS", config.evaluation.evaluate_every_n_rounds)
    _set(
        env,
        "EVAL_METRICS",
        [metric.value for metric in config.evaluation.metrics],
        as_json=True,
    )

    _set(env, "ARTIFACT_DIR", resolved_artifact_dir)
    _set(env, "CKPT_EVERY_N_ROUNDS", config.artifacts.checkpoint_every_n_rounds)
    _set(env, "SAVE_FINAL_MODEL", config.artifacts.save_final_model, as_bool=True)
    _set(env, "SAVE_OPT_STATE", config.artifacts.save_optimizer_state, as_bool=True)

    _set(env, "FLOWERTUNE_LOG_LEVEL", config.logging.log_level.value)
    _set(
        env,
        "REPORT_ROUND_METRICS",
        config.logging.report_round_metrics,
        as_bool=True,
    )
    _set(
        env,
        "REPORT_CLIENT_METRICS",
        config.logging.report_client_metrics,
        as_bool=True,
    )

    _set(env, "ROUND_TIMEOUT_SEC", config.timeouts.round_timeout_sec)
    _set(env, "JOB_TIMEOUT_SEC", config.timeouts.job_timeout_sec)
    return env
