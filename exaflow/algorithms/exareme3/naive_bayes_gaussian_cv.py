import numpy as np

from exaflow.algorithms.exareme3.naive_bayes_common import NBResult
from exaflow.algorithms.exareme3.naive_bayes_common import make_naive_bayes_result
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.cross_validation import FederatedCrossValidator
from exaflow.algorithms.federated.cross_validation import FederatedKFoldSplitter
from exaflow.algorithms.federated.cross_validation import (
    FederatedMulticlassClassificationScorer,
)
from exaflow.algorithms.federated.cross_validation.scorer_multiclass import (
    multiclass_classification_metrics,
)
from exaflow.algorithms.federated.cross_validation.scorer_multiclass import (
    multiclass_classification_summary,
)
from exaflow.algorithms.federated.naive_bayes_gaussian import FederatedGaussianNB
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "naive_bayes_gaussian_cv"
VAR_SMOOTHING = 1e-9  # same as original GaussianNB _fit_global


class GaussianNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self) -> NBResult:
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        n_splits = self.get_parameter("n_splits")

        # Sorted class labels to match sklearn/original implementation
        label_dict = self.metadata[y_var]["enumerations"]
        labels = sorted(label_dict.keys())

        # Run CV UDF (with aggregation server)
        udf_results = self.run_local_udf(
            func=gaussian_nb_cv_local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
                "n_splits": int(n_splits),
            },
        )

        metrics = udf_results[0]  # identical on all workers

        confmats = [np.asarray(cm, dtype=float) for cm in metrics["confmats"]]
        n_obs = [int(v) for v in metrics["n_obs"]]

        # Aggregate confusion matrix across folds
        total_confmat = (
            sum(confmats)
            if confmats
            else np.zeros((len(labels), len(labels)), dtype=float)
        )

        # Compute per-fold metrics and summary, using original helpers
        per_fold_metrics = [
            multiclass_classification_metrics(confmat) for confmat in confmats
        ]
        summary = multiclass_classification_summary(per_fold_metrics, labels, n_obs)

        return make_naive_bayes_result(total_confmat, labels, summary)


@exareme3_udf(with_aggregation_server=True)
def gaussian_nb_cv_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    labels,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for Gaussian Naive Bayes.
    """
    n_splits = int(n_splits)
    class_labels = list(labels)
    if not class_labels:
        return {"confmats": [], "n_obs": []}

    valid_mask = data[y_var].notna().to_numpy()
    data_valid = data.loc[valid_mask]
    n_rows = int(data_valid.shape[0])
    if n_rows == 0 or n_rows < n_splits:
        raise BadInputError(
            "Cross validation cannot run because the number of observations "
            f"({n_rows}) is smaller than the number of splits ({n_splits})."
        )

    X = data_valid[x_vars].to_numpy(dtype=float, copy=False)
    y = data_valid[y_var].to_numpy()

    splitter = FederatedKFoldSplitter(n_splits=n_splits, shuffle=False)
    estimator = FederatedGaussianNB(
        x_vars=x_vars,
        labels=class_labels,
        var_smoothing=VAR_SMOOTHING,
    )
    scorer = FederatedMulticlassClassificationScorer(labels=class_labels)
    cross_validator = FederatedCrossValidator(
        estimator=estimator,
        splitter=splitter,
        scorer=scorer,
    )

    try:
        metrics = cross_validator.evaluate(X, y, agg_client=agg_client)
    except BadInputError as exc:
        raise BadUserInput(str(exc))

    confmats_global = [np.asarray(cm, dtype=float) for cm in metrics["confmat"]]
    n_obs_per_fold = [int(v) for v in metrics["n_obs"]]

    return {
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }
