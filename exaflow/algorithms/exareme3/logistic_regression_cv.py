import statistics as stats
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import sklearn.metrics as skm
from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.model_selection.cross_validation.cross_validator import (
    FederatedCrossValidator,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_classification import (
    FederatedClassificationScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_classification import (
    compute_classification_metrics_from_confmat,
)
from exaflow.algorithms.federated.model_selection.cross_validation.splitter_kfold import (
    FederatedKFoldSplitter,
)
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from exaflow.algorithms.federated.preprocessing import FederatedPassthrough
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.specifications import AlgorithmName
from exaflow.worker_communication import BadUserInput


class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            fn=self.fn + other.fn,
        )

    def ravel(self):
        # Backwards-compatible with original code: tn, fp, fn, tp = confmat.ravel()
        return [self.tn, self.fp, self.fn, self.tp]


class CVClassificationSummary(BaseModel):
    row_names: List[str]
    n_obs: List[Optional[int]]
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    fscore: List[float]


class ROCCurve(BaseModel):
    name: str
    tpr: List[float]
    fpr: List[float]
    auc: float


class CVLogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: CVClassificationSummary
    confusion_matrix: ConfusionMatrix
    roc_curves: List[ROCCurve]


class BasicMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    fscore: float


class LogisticRegressionCV(Algorithm, algname=AlgorithmName.LOGISTIC_REGRESSION_CV):
    def run(self):
        positive_class = self.get_parameter("positive_class")
        n_splits = self.get_parameter("n_splits")
        y_var = self.inputdata.y[0]

        categorical_vars = [
            var for var in self.inputdata.x if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not self.metadata[var]["is_categorical"]
        ]

        # Run distributed logistic CV
        udf_results = self.run_local_udf(
            func=logistic_regression_cv_local_step,
            kw_args={
                "y_var": y_var,
                "positive_class": positive_class,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "n_splits": n_splits,
            },
        )

        metrics = udf_results[0]
        indep_var_names = metrics["feature_names"]

        n_obs_train = [int(v) for v in metrics["n_obs"]]
        tp_list = [int(v) for v in metrics["tp"]]
        fp_list = [int(v) for v in metrics["fp"]]
        tn_list = [int(v) for v in metrics["tn"]]
        fn_list = [int(v) for v in metrics["fn"]]
        roc_tpr = metrics["roc_tpr"]  # list of list
        roc_fpr = metrics["roc_fpr"]  # list of list

        # Per-fold confusion matrices
        fold_confmats = [
            ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
            for tp, fp, tn, fn in zip(tp_list, fp_list, tn_list, fn_list)
        ]

        # Total confusion matrix over all folds
        total_confmat = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        for cm in fold_confmats:
            total_confmat += cm

        summary = make_classification_metrics_summary(
            n_splits=n_splits, n_obs=n_obs_train, confmats=fold_confmats
        )

        # ROC curves per fold + AUC
        roc_curves_result: List[ROCCurve] = []
        for i, (tpr, fpr) in enumerate(zip(roc_tpr, roc_fpr)):
            auc_val = float(skm.auc(x=fpr, y=tpr)) if len(tpr) > 1 else 0.0
            roc_curves_result.append(
                ROCCurve(
                    name=f"fold_{i + 1}",
                    tpr=tpr,
                    fpr=fpr,
                    auc=auc_val,
                )
            )

        dependent_var = y_var
        return CVLogisticRegressionResult(
            dependent_var=dependent_var,
            indep_vars=indep_var_names,
            summary=summary,
            confusion_matrix=total_confmat,
            roc_curves=roc_curves_result,
        )


@exareme3_udf(with_aggregation_server=True)
def logistic_regression_cv_local_step(
    agg_client,
    data,
    y_var,
    positive_class,
    categorical_vars,
    numerical_vars,
    n_splits,
):
    """
    Run K-fold CV for logistic regression using secure aggregation.

    For each fold:
    - Train a global logistic model via federated aggregation.
    - Compute probabilities on the test set.
    - Aggregate confusion-matrix counts (threshold 0.5).
    - Approximate ROC curve on a fixed grid of thresholds via aggregated counts.
    """
    n_splits = int(n_splits)

    cv_pipeline = FederatedPipeline(
        [
            (
                "features",
                FederatedColumnTransformer(
                    [
                        ("cat", FederatedOneHotEncoder(), "categorical"),
                        ("num", FederatedPassthrough(), "numerical"),
                    ]
                ),
            ),
            ("model", FederatedLogisticRegression(fit_intercept=True)),
        ]
    )
    positive_class = FederatedLogisticRegression.coerce_positive_class(
        data[y_var], positive_class
    )
    y = data[y_var].eq(positive_class).astype(float).to_numpy()

    # Fixed grid of thresholds for ROC approximation
    thresholds = np.linspace(0.0, 1.0, 101)

    splitter = FederatedKFoldSplitter(n_splits=n_splits, shuffle=False)
    scorer = FederatedClassificationScorer(thresholds=thresholds)
    cross_validator = FederatedCrossValidator(
        estimator=cv_pipeline,
        splitter=splitter,
        scorer=scorer,
    )

    try:
        metrics = cross_validator.evaluate(
            None,
            y,
            data=data,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
            agg_client=agg_client,
        )
    except BadInputError as exc:
        raise BadUserInput(str(exc))

    # Get global feature names
    feature_transformer = FederatedColumnTransformer(
        [
            ("cat", FederatedOneHotEncoder(), "categorical"),
            ("num", FederatedPassthrough(), "numerical"),
        ]
    )
    feature_transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = feature_transformer.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = ["Intercept"] + feature_names

    return {
        "n_obs": metrics["n_obs"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "roc_tpr": metrics["roc_tpr"],
        "roc_fpr": metrics["roc_fpr"],
        "feature_names": feature_names,
    }


def make_classification_metrics_summary(
    n_splits: int, n_obs: List[int], confmats: List[ConfusionMatrix]
) -> CVClassificationSummary:
    row_names = [f"fold_{i}" for i in range(1, n_splits + 1)] + ["average", "stdev"]

    metrics = []
    for confmat in confmats:
        stats_dict = compute_classification_metrics_from_confmat(
            {
                "tp": confmat.tp,
                "fp": confmat.fp,
                "tn": confmat.tn,
                "fn": confmat.fn,
            }
        )
        metrics.append(
            BasicMetrics(
                accuracy=stats_dict["accuracy"],
                precision=stats_dict["precision"],
                recall=stats_dict["recall"],
                fscore=stats_dict["fscore"],
            )
        )

    accuracy, precision, recall, fscore = zip(*metrics)

    accuracy = list(accuracy) + [stats.mean(accuracy), stats.stdev(accuracy)]
    precision = list(precision) + [stats.mean(precision), stats.stdev(precision)]
    recall = list(recall) + [stats.mean(recall), stats.stdev(recall)]
    fscore = list(fscore) + [stats.mean(fscore), stats.stdev(fscore)]

    return CVClassificationSummary(
        row_names=row_names,
        n_obs=n_obs + [None, None],  # we don't compute average & stderr for n_obs
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )
