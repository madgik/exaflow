from typing import Dict
from typing import List

import numpy as np

from exaflow.algorithms.exareme3.naive_bayes_common import make_naive_bayes_result
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.model_selection.cross_validation import (
    FederatedCrossValidator,
)
from exaflow.algorithms.federated.model_selection.cross_validation import (
    FederatedKFoldSplitter,
)
from exaflow.algorithms.federated.model_selection.cross_validation import (
    FederatedMulticlassClassificationScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_multiclass import (
    multiclass_classification_metrics,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_multiclass import (
    multiclass_classification_summary,
)
from exaflow.algorithms.federated.naive_bayes import FederatedCategoricalNB
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOrdinalEncoder
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "naive_bayes_categorical_cv"


class NaiveBayesCategoricalAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        n_splits = self.get_parameter("n_splits")

        # Build sorted category lists to match sklearn / original implementation
        all_vars = x_vars + [y_var]
        categories: Dict[str, List[str]] = {
            var: list(sorted(self.metadata[var]["enumerations"].keys()))
            for var in all_vars
        }

        udf_results = self.run_local_udf(
            func=naive_bayes_categorical_cv_local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
                "n_splits": int(n_splits),
            },
        )

        metrics = udf_results[0]  # identical on all workers

        labels = metrics["labels"]
        confmats = [np.asarray(cm, dtype=float) for cm in metrics["confmats"]]
        n_obs = [int(v) for v in metrics["n_obs"]]

        # Aggregate across folds using the original helpers
        total_confmat = sum(confmats)  # element-wise sum
        per_fold_metrics = [
            multiclass_classification_metrics(confmat) for confmat in confmats
        ]
        summary = multiclass_classification_summary(per_fold_metrics, labels, n_obs)

        # Use the same helper as the Gaussian NB CV to build the final result
        result = make_naive_bayes_result(total_confmat, labels, summary)
        return result


@exareme3_udf(with_aggregation_server=True)
def naive_bayes_categorical_cv_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for categorical
    Naive Bayes with secure aggregation.
    """

    n_splits = int(n_splits)

    labels = list(categories[y_var])
    if not labels:
        return {
            "labels": [],
            "confmats": [],
            "n_obs": [],
        }

    valid_mask = data[y_var].notna().to_numpy()
    data_valid = data.loc[valid_mask]
    n_rows = int(data_valid.shape[0])
    if n_rows == 0 or n_rows < n_splits:
        raise BadInputError(
            "Cross validation cannot run because the number of observations "
            f"({n_rows}) is smaller than the number of splits ({n_splits})."
        )
    y = data_valid[y_var].to_numpy()

    splitter = FederatedKFoldSplitter(n_splits=n_splits, shuffle=False)
    estimator = FederatedPipeline(
        [
            (
                "features",
                FederatedOrdinalEncoder(
                    categories=categories,
                    handle_unknown="error",
                ),
            ),
            (
                "model",
                FederatedCategoricalNB(
                    y_var=y_var,
                    x_vars=x_vars,
                    categories=categories,
                ),
            ),
        ]
    )
    scorer = FederatedMulticlassClassificationScorer(labels=labels)
    cross_validator = FederatedCrossValidator(
        estimator=estimator,
        splitter=splitter,
        scorer=scorer,
    )

    try:
        metrics = cross_validator.evaluate(
            None,
            y,
            data=data_valid,
            categorical_vars=x_vars,
            numerical_vars=[],
            agg_client=agg_client,
        )
    except BadInputError as exc:
        raise BadUserInput(str(exc))
    confmats_global = [np.asarray(cm, dtype=float) for cm in metrics["confmat"]]
    n_obs_per_fold = [int(v) for v in metrics["n_obs"]]

    return {
        "labels": labels,
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }
