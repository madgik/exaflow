"""MetricsOut validation for flowertune_llm_medical final summary payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError


class _ErrorInfo(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] | None = None


class _Artifacts(BaseModel):
    final_checkpoint_ref: str = Field(..., min_length=1)
    intermediate_checkpoints: List[str] | None = None


class _RoundArtifacts(BaseModel):
    checkpoint_ref: str | None = None


class _RoundClients(BaseModel):
    participated: int = Field(..., ge=0)
    expected: int = Field(..., ge=1)
    failed: int = Field(0, ge=0)


class _RoundEvent(BaseModel):
    schema_version: str
    payload_type: str
    job_id: str
    algorithm_name: str
    runtime: str
    status: str
    round: int = Field(..., ge=1)
    rounds_total: int = Field(..., ge=1)
    timestamp: str
    requested_metrics: List[str]
    reported_metrics: List[str]
    metrics: Dict[str, float]
    clients: _RoundClients
    artifacts: _RoundArtifacts
    error: _ErrorInfo | None = None


class _FinalSummary(BaseModel):
    schema_version: str
    payload_type: str
    job_id: str
    algorithm_name: str
    runtime: str
    status: str
    timestamp: str
    rounds_total: int
    rounds_completed: int
    requested_metrics: List[str]
    reported_metrics: List[str]
    aggregate_metrics: Dict[str, float]
    artifacts: _Artifacts | None = None
    error: _ErrorInfo | None = None


def _is_utc_z(ts: str) -> bool:
    if not ts.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_final_summary_payload(payload: Dict[str, Any]) -> None:
    """Validate final summary payload against Phase 1.1 policy."""

    try:
        parsed = _FinalSummary.parse_obj(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid final_summary payload shape: {exc}") from exc

    if parsed.schema_version != "1.1":
        raise ValueError("final_summary.schema_version must be '1.1'")
    if parsed.payload_type != "final_summary":
        raise ValueError("payload_type must be 'final_summary'")
    if parsed.runtime != "simulation":
        raise ValueError("runtime must be 'simulation'")
    if parsed.status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
        raise ValueError("invalid final_summary status")
    if not _is_utc_z(parsed.timestamp):
        raise ValueError("timestamp must be UTC and end with 'Z'")

    if parsed.rounds_completed > parsed.rounds_total:
        raise ValueError("rounds_completed must be <= rounds_total")

    metric_keys = sorted(parsed.aggregate_metrics.keys())
    if parsed.status == "COMPLETED":
        if parsed.error is not None:
            raise ValueError("error must be omitted for COMPLETED status")
        if parsed.artifacts is None:
            raise ValueError("artifacts are required for COMPLETED status")
        if sorted(parsed.reported_metrics) != metric_keys:
            raise ValueError(
                "reported_metrics must equal aggregate_metrics keys for COMPLETED status"
            )
    else:
        if parsed.error is None:
            raise ValueError("error is required for non-COMPLETED status")
        if parsed.aggregate_metrics and not set(parsed.reported_metrics).issubset(
            parsed.aggregate_metrics.keys()
        ):
            raise ValueError(
                "reported_metrics must be subset of aggregate_metrics keys for non-COMPLETED status"
            )


def validate_round_event_payload(payload: Dict[str, Any]) -> None:
    """Validate round event payload against Phase 1.1 policy."""

    try:
        parsed = _RoundEvent.parse_obj(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid round_event payload shape: {exc}") from exc

    if parsed.schema_version != "1.1":
        raise ValueError("round_event.schema_version must be '1.1'")
    if parsed.payload_type != "round_event":
        raise ValueError("payload_type must be 'round_event'")
    if parsed.runtime != "simulation":
        raise ValueError("runtime must be 'simulation'")
    if parsed.status not in {"RUNNING", "FAILED"}:
        raise ValueError("round_event.status must be RUNNING or FAILED")
    if not _is_utc_z(parsed.timestamp):
        raise ValueError("timestamp must be UTC and end with 'Z'")
    if parsed.round > parsed.rounds_total:
        raise ValueError("round must be <= rounds_total")
    if parsed.clients.participated > parsed.clients.expected:
        raise ValueError("clients.participated must be <= clients.expected")
    if parsed.clients.failed > parsed.clients.expected:
        raise ValueError("clients.failed must be <= clients.expected")
    if (parsed.clients.participated + parsed.clients.failed) > parsed.clients.expected:
        raise ValueError(
            "clients.participated + clients.failed must be <= clients.expected"
        )

    if parsed.status == "RUNNING":
        if parsed.error is not None:
            raise ValueError("error must be omitted for RUNNING round_event")
        if sorted(parsed.reported_metrics) != sorted(parsed.metrics.keys()):
            raise ValueError("reported_metrics must equal metrics keys for RUNNING")
    else:
        if parsed.error is None:
            raise ValueError("error is required for FAILED round_event")
        if parsed.metrics and not set(parsed.reported_metrics).issubset(
            parsed.metrics.keys()
        ):
            raise ValueError(
                "reported_metrics must be subset of metrics keys for FAILED round_event"
            )
