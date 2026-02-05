from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class TTestResult(BaseModel):
    t_stat: float
    df: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str
    ci_lower: str
    cohens_d: float


class OneSampleTTestResult(TTestResult):
    n_obs: int
    std: float


def _common_result_fields(raw: Dict) -> Dict:
    return {
        "t_stat": raw["t_stat"],
        "df": raw["df"],
        "p": raw["p_value"],
        "mean_diff": raw["mean_diff"],
        "se_diff": raw["se_diff"],
        "ci_upper": raw["ci_upper"],
        "ci_lower": raw["ci_lower"],
        "cohens_d": raw["cohens_d"],
    }


def build_basic_ttest_result(raw: Dict) -> TTestResult:
    return TTestResult(**_common_result_fields(raw))


def build_one_sample_ttest_result(raw: Dict) -> OneSampleTTestResult:
    fields = {
        "n_obs": raw["n_obs"],
        "std": raw["std"],
    }
    fields.update(_common_result_fields(raw))
    return OneSampleTTestResult(**fields)
