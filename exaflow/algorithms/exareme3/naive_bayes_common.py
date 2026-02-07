import typing as t
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ConfusionMatrix(BaseModel):
    """Multiclass confusion matrix model

    Each row of the matrix represents the instances in an actual class while
    each column represents the instances in a predicted class.

    Attributes
    ----------
    data
        Confusion matrix data in row major order
    labels
        Labels of the classes used in classification
    """

    data: t.List[t.List[int]]
    labels: t.List[str]


class MulticlassClassificationSummary(BaseModel):
    """Multiclass classification summary model

    In cross validated multiclass classification, the accuracy, precision,
    recall and fscore are computed for every class, for every fold. The number
    of observations, n_obs, is different for every fold, but doesn't depend on
    the class.

    This produces a hierarchical table. E.g. for two classes cl1, cl2 the table
    has the following form.

    |      | accuracy  | precision |  recall   |  fscore   |       |
    | fold | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | n_obs |
    |------+-----------+-----------+-----------+-----------|-------|
    |    1 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |
    |    2 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |

    This table is represented as a collection of mappings. For the hierarchical
    quantities these mappings are nested and have the form
        {"accuracy": {"cl1": ..., "cl2": ...}, ...}
    """

    accuracy: t.Dict[str, t.Dict[str, float]]
    precision: t.Dict[str, t.Dict[str, float]]
    recall: t.Dict[str, t.Dict[str, float]]
    fscore: t.Dict[str, t.Dict[str, float]]
    n_obs: t.Dict[str, int]


class NBResult(BaseModel):
    confusion_matrix: ConfusionMatrix
    classification_summary: MulticlassClassificationSummary


def make_naive_bayes_result(confmat, labels, summary) -> NBResult:
    """Helper to build the NBResult from a confusion matrix + summary dict."""
    confmat_model = ConfusionMatrix(data=confmat.tolist(), labels=labels)
    summary_model = MulticlassClassificationSummary(**summary)
    result = NBResult(
        confusion_matrix=confmat_model,
        classification_summary=summary_model,
    )
    return result
