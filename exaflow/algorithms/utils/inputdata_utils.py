from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set

import pandas as pd
from pydantic import BaseModel

from exaflow.column_names import DATASET_COL


class Inputdata(BaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]] = None
    filters: Optional[dict] = None
    variables: List[str]


def _apply_filter(df: pd.DataFrame, filter_rule: Dict[str, Any]) -> pd.DataFrame:
    """
    Keep the rows of ``df`` that satisfy a jQuery QueryBuilder filter.

    Rules are evaluated as boolean masks so OR keeps row order and does not
    collapse genuinely identical rows.
    """
    return df[_filter_mask(df, filter_rule)]


_OPERATOR_MASKS = {
    "equal": lambda col, value: col == value,
    "not_equal": lambda col, value: col != value,
    "greater": lambda col, value: col > value,
    "less": lambda col, value: col < value,
    "greater_or_equal": lambda col, value: col >= value,
    "less_or_equal": lambda col, value: col <= value,
    "between": lambda col, value: col.between(value[0], value[1]),
    "not_between": lambda col, value: ~col.between(value[0], value[1]),
    "is_not_null": lambda col, value: col.notnull(),
    "is_null": lambda col, value: col.isnull(),
    "in": lambda col, value: col.isin(value),
    "not_in": lambda col, value: ~col.isin(value),
}


def _filter_mask(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
    if "condition" in rule:
        masks = [_filter_mask(df, sub_rule) for sub_rule in rule.get("rules", [])]
        condition = rule["condition"]
        if condition == "AND":
            combined = pd.Series(True, index=df.index)
            for mask in masks:
                combined &= mask
            return combined
        if condition == "OR":
            combined = pd.Series(False, index=df.index)
            for mask in masks:
                combined |= mask
            return combined
        raise ValueError(f"Unsupported condition: {condition}")

    operator = rule["operator"]
    if operator not in _OPERATOR_MASKS:
        raise ValueError(f"Unsupported operator: {operator}")
    return _OPERATOR_MASKS[operator](df[rule["id"]], rule.get("value"))


def _filter_columns(filter_rule: Optional[Dict[str, Any]]) -> Set[str]:
    if not filter_rule:
        return set()
    if "condition" in filter_rule:
        columns: Set[str] = set()
        for sub_rule in filter_rule.get("rules", []):
            columns |= _filter_columns(sub_rule)
        return columns
    return {filter_rule["id"]}


def _apply_inputdata(
    df: pd.DataFrame, inputdata: Inputdata, dropna: bool
) -> pd.DataFrame:
    """
    Filters the dataframe based on the provided inputdata including filters, datasets, and selected columns.
    """
    if inputdata.filters:
        df = _apply_filter(df, inputdata.filters)

    # Filter based on the provided datasets.
    all_datasets = inputdata.datasets + (
        inputdata.validation_datasets if inputdata.validation_datasets else []
    )
    df = df[df[DATASET_COL].isin(all_datasets)]

    # Remove duplicates while preserving order to mirror the behaviour of the
    # SQL views used in the original algorithms.
    columns = list(dict.fromkeys(inputdata.variables))

    if not columns:
        raise ValueError("'variables' are missing or empty in inputdata.")

    dataset_column = DATASET_COL
    select_columns = columns + (
        [dataset_column] if dataset_column in df.columns else []
    )
    # Select only the required columns (keep dataset column if available for downstream grouping).
    df = df[select_columns]
    if dropna:
        # Drop rows with missing values in any of the requested columns.
        # This matches the SQL `IS NOT NULL` filters that the non-exaflow algorithms use.
        df = df.dropna(subset=columns)

    return df


def _read_filtered_chunks(
    path: str,
    needed_columns: Set[str],
    inputdata: Inputdata,
    dropna: bool,
) -> Iterator[pd.DataFrame]:
    """
    Yields filtered DataFrame chunks from a CSV file.
    """
    for chunk in pd.read_csv(path, usecols=needed_columns, chunksize=10000):
        filtered_chunk = _apply_inputdata(chunk, inputdata, dropna)
        if not filtered_chunk.empty:
            yield filtered_chunk


def fetch_data(
    inputdata: Inputdata, csv_paths: List[str], *, dropna: bool = True
) -> pd.DataFrame:
    """
    Loads CSV data from the given paths in chunks, applies filtering based on inputdata,
    and concatenates the results into a single DataFrame.

    Args:
        inputdata: An instance of Inputdata with filter and column selection instructions.
        csv_paths: A list of CSV file paths.

    Returns:
        A concatenated DataFrame after filtering.
    """
    # Include "dataset" and every filtered column so filters can be evaluated
    needed_columns = set(inputdata.variables + ["dataset"]) | _filter_columns(
        inputdata.filters
    )

    # Gather filtered chunks from all CSV files
    chunks = []
    for path in set(csv_paths):
        chunks.extend(
            list(_read_filtered_chunks(path, needed_columns, inputdata, dropna))
        )

    return (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=list(needed_columns))
    )
