#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

DEFAULT_BASE_URL = "http://127.0.0.1:5100"
DEFAULT_DATASETS = [f"edsd{idx}" for idx in range(10)]
DEFAULT_CLUSTER_VARIABLES = ["lefthippocampus", "righthippocampus"]
DEFAULT_DOWNSTREAM_Y = "leftamygdala"
DEFAULT_DOWNSTREAM_X = ["kmeans_cluster"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args)
    if args.list_cases:
        for case in cases:
            print(case["case_id"])
        return 0

    if not healthcheck(args.base_url):
        print(
            f"ERROR: Exaflow API is not reachable at {args.base_url}. "
            "Run `uv run inv deploy` first and retry.",
            file=sys.stderr,
        )
        return 2

    if args.cases:
        requested_cases = set(args.cases)
        known_cases = {case["case_id"] for case in cases}
        unknown_cases = sorted(requested_cases - known_cases)
        if unknown_cases:
            print(
                "ERROR: Unknown case(s): "
                + ", ".join(unknown_cases)
                + "\nKnown cases: "
                + ", ".join(sorted(known_cases)),
                file=sys.stderr,
            )
            return 2
        cases = [case for case in cases if case["case_id"] in requested_cases]

    summaries = []
    for case in cases:
        print(f"Running {case['case_id']} ...")
        response = post_json(f"{args.base_url}/analysis", case["payload"])
        case_path = output_dir / f"{case['case_id']}.json"
        case_path.write_text(
            json.dumps(response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries.append(summarize_case(case, response, case_path))

    summary_path = output_dir / "summary.md"
    summary_path.write_text(render_summary(summaries), encoding="utf-8")
    print(f"\nWrote {len(summaries)} result files to {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local KMeans experiment variants against Exaflow API.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--data-model", default="dementia:0.1")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--cluster-variables",
        nargs="+",
        default=DEFAULT_CLUSTER_VARIABLES,
        help="Numerical variables used to fit KMeans.",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument(
        "--run",
        choices=["report", "preprocessing", "all"],
        default="all",
        help="Which experiment family to run.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Run only specific case ids. Use --list-cases to print available cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available case ids for the selected --run mode and exit.",
    )
    parser.add_argument(
        "--preprocessing-modes",
        nargs="+",
        choices=["full", "binary", "subset"],
        default=["full", "binary", "subset"],
    )
    parser.add_argument(
        "--binary-cluster",
        default="cluster_1",
        help="Cluster used for binary preprocessing mode.",
    )
    parser.add_argument(
        "--selected-clusters",
        nargs="+",
        default=["cluster_1", "cluster_2"],
        help="Clusters kept for subset preprocessing mode.",
    )
    parser.add_argument(
        "--downstream-y",
        default=DEFAULT_DOWNSTREAM_Y,
        help="Y variable for downstream linear regression preprocessing experiments.",
    )
    parser.add_argument(
        "--downstream-x",
        nargs="+",
        default=DEFAULT_DOWNSTREAM_X,
        help="X variables for downstream linear regression. Include kmeans_cluster to test the generated covariate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to .kmeans_experiments/<timestamp>.",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(".kmeans_experiments") / timestamp)
    return args


def healthcheck(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/healthcheck", timeout=10) as response:
            return 200 <= response.status < 300
    except URLError:
        return False


def build_cases(args: argparse.Namespace) -> list[dict]:
    cases = []
    if args.run in {"report", "all"}:
        cases.extend(build_report_cases(args))
    if args.run in {"preprocessing", "all"}:
        cases.extend(build_preprocessing_cases(args))
    return cases


def build_report_cases(args: argparse.Namespace) -> list[dict]:
    variants = [
        (
            "report_manual_random_range",
            {
                "k_selection": "manual",
                "k": args.k,
                "init_method": "random_range",
                "n_init": args.n_init,
            },
        ),
        (
            "report_manual_multi_start",
            {
                "k_selection": "manual",
                "k": args.k,
                "init_method": "multi_start_random_range",
                "n_init": args.n_init,
            },
        ),
        (
            "report_elbow_random_range",
            {
                "k_selection": "elbow",
                "k_min": args.k_min,
                "k_max": args.k_max,
                "init_method": "random_range",
                "n_init": args.n_init,
            },
        ),
        (
            "report_elbow_multi_start",
            {
                "k_selection": "elbow",
                "k_min": args.k_min,
                "k_max": args.k_max,
                "init_method": "multi_start_random_range",
                "n_init": args.n_init,
            },
        ),
    ]
    return [
        {
            "case_id": case_id,
            "kind": "report",
            "payload": analysis_payload(
                data_model=args.data_model,
                datasets=args.datasets,
                variables=args.cluster_variables,
                preprocessing=[
                    missing_values_step(args.cluster_variables),
                ],
                algorithm={
                    "name": "kmeans",
                    "x": None,
                    "y": args.cluster_variables,
                    "parameters": {
                        **params,
                        "tol": args.tol,
                        "maxiter": args.maxiter,
                    },
                },
            ),
        }
        for case_id, params in variants
    ]


def build_preprocessing_cases(args: argparse.Namespace) -> list[dict]:
    cases = []
    for mode in args.preprocessing_modes:
        params = {
            "code": "kmeans_cluster",
            "cluster_variables": args.cluster_variables,
            "k_selection": "manual",
            "k": args.k,
            "tol": args.tol,
            "maxiter": args.maxiter,
            "init_method": "multi_start_random_range",
            "n_init": args.n_init,
            "output_mode": mode,
        }
        if mode == "binary":
            params["binary_cluster"] = args.binary_cluster
        if mode == "subset":
            params["selected_clusters"] = args.selected_clusters

        input_variables = unique_list(
            args.cluster_variables
            + [args.downstream_y]
            + [
                variable
                for variable in args.downstream_x
                if variable != "kmeans_cluster"
            ]
        )
        missing_value_variables = unique_list(
            args.cluster_variables + [args.downstream_y]
        )
        cases.append(
            {
                "case_id": f"preprocessing_{mode}_linear_regression",
                "kind": "preprocessing",
                "payload": analysis_payload(
                    data_model=args.data_model,
                    datasets=args.datasets,
                    variables=input_variables,
                    preprocessing=[
                        missing_values_step(missing_value_variables),
                        {
                            "name": "kmeans_cluster_creator",
                            "parameters": params,
                        },
                    ],
                    algorithm={
                        "name": "linear_regression",
                        "x": args.downstream_x,
                        "y": [args.downstream_y],
                        "parameters": {},
                    },
                ),
            }
        )
    return cases


def analysis_payload(
    *,
    data_model: str,
    datasets: list[str],
    variables: list[str],
    algorithm: dict,
    preprocessing: list[dict] | None = None,
) -> dict:
    return {
        "inputdata": {
            "data_model": data_model,
            "datasets": datasets,
            "filters": None,
            "variables": variables,
        },
        "preprocessing": preprocessing,
        "algorithm": algorithm,
    }


def missing_values_step(variables: list[str]) -> dict:
    return {
        "name": "missing_values_handler",
        "parameters": {
            "strategies": {variable: "drop" for variable in variables},
        },
    }


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8")
            return parse_response_text(text)
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {
            "error": True,
            "status": exc.code,
            "body": text,
        }
    except URLError as exc:
        return {
            "error": True,
            "status": None,
            "body": str(exc),
        }


def parse_response_text(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": True, "status": 200, "body": text}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def summarize_case(case: dict, response: dict, case_path: Path) -> dict:
    if response.get("error"):
        return {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "status": "ERROR",
            "selected_k": "",
            "init": "",
            "n_init": "",
            "warnings": response.get("body", ""),
            "file": str(case_path),
        }
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "status": "OK",
        "selected_k": response.get("selected_k", ""),
        "init": response.get("initialization_method", ""),
        "n_init": response.get("n_init", ""),
        "warnings": "; ".join(response.get("warnings", []) or []),
        "file": str(case_path),
    }


def render_summary(summaries: list[dict]) -> str:
    lines = [
        "# KMeans Experiments",
        "",
        "| case | kind | status | selected_k | init | n_init | warnings | file |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {case_id} | {kind} | {status} | {selected_k} | {init} | "
            "{n_init} | {warnings} | {file} |".format(
                **{key: markdown_cell(value) for key, value in summary.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def markdown_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def unique_list(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
