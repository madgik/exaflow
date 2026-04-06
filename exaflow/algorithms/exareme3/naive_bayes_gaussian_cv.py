import numpy as np

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.naive_bayes_common import NBResult
from exaflow.algorithms.exareme3.naive_bayes_common import make_naive_bayes_result
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
from exaflow.algorithms.federated.naive_bayes import FederatedGaussianNB
from exaflow.algorithms.federated.utils import BadInputError

VAR_SMOOTHING = 1e-9  # same as original GaussianNB _fit_global


class NaiveBayesGaussianCV(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="naive_bayes_gaussian_cv",
            desc="Federated Gaussian Naive Bayes with K-fold cross-validation. Features are treated as numerical; missing values are not imputed.",
            label="Gaussian Naive Bayes (K-fold CV)",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable (dependent)",
                    desc="A unique nominal variable.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates (independent)",
                    desc="One or more numerical variables.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "n_splits": specs.ParameterSpecification(
                    label="Number of splits",
                    desc="Number of splits for cross-validation.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=5,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=2,
                    max=20,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self) -> NBResult:
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        n_splits = self.get_parameter("n_splits")
        labels = sorted(get_enum_codes(self.metadata, y_var))

        # Run CV UDF (with aggregation server)
        metrics = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
                "n_splits": int(n_splits),
            },
            identical_results=True,
        )
        labels = metrics["labels"]

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
def local_step(
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
        return {"labels": [], "confmats": [], "n_obs": []}

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

    metrics = cross_validator.evaluate(X, y, agg_client=agg_client)

    confmats_global = [np.asarray(cm, dtype=float) for cm in metrics["confmat"]]
    n_obs_per_fold = [int(v) for v in metrics["n_obs"]]

    return {
        "labels": class_labels,
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }
