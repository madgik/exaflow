from typing import Dict
from typing import List

import numpy as np

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.naive_bayes.naive_bayes_common import (
    make_naive_bayes_result,
)
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
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


class NaiveBayesCategorical(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="naive_bayes_categorical_cv",
            desc="Categorical Naive Bayes evaluated with K-fold cross-validation.",
            documentation=(
                "Evaluates a categorical Naive Bayes classifier with K-fold "
                "cross-validation for a nominal outcome and nominal features. "
                "Features are ordinal-encoded using metadata category order; "
                "unknown categories are rejected.\n\n"
                "The 'n_splits' setting controls the number of cross-validation "
                "folds. It must be between 2 and 20. Default is 5.\n\n"
                "The result includes multiclass classification metrics and a "
                "summary across folds.\n\n"
                "Reference behavior is aligned with scikit-learn CategoricalNB "
                "and KFold cross-validation methodology, using aggregated "
                "class/category counts and confusion matrices without sharing "
                "raw data."
            ),
            label="Categorical Naive Bayes (K-fold CV)",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Nominal outcome variable.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Features",
                desc="Nominal features used for classification.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
            ),
            parameters={
                "n_splits": specs.ParameterSpecification(
                    label="Number of folds",
                    desc="Fold count used for cross-validation.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=5,
                    min=2,
                    max=20,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.y[0]
        x_vars = list(self.x)
        n_splits = self.get_parameter("n_splits")
        categories: Dict[str, List[str]] = {
            var: sorted(get_enum_codes(self.metadata, var)) for var in x_vars + [y_var]
        }

        metrics = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
                "n_splits": int(n_splits),
            },
            identical_results=True,
        )

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
def local_step(
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

    metrics = cross_validator.evaluate(
        None,
        y,
        data=data_valid,
        categorical_vars=x_vars,
        numerical_vars=[],
        agg_client=agg_client,
    )
    confmats_global = [np.asarray(cm, dtype=float) for cm in metrics["confmat"]]
    n_obs_per_fold = [int(v) for v in metrics["n_obs"]]

    return {
        "labels": labels,
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }
