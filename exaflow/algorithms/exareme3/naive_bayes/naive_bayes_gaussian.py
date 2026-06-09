from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes import FederatedGaussianNB


class NaiveBayesGaussianResult(BaseModel):
    classes: List[str]
    class_count: List[float]
    class_prior: List[float]
    theta: List[List[float]]
    var: List[List[float]]
    feature_names: List[str]


class NaiveBayesGaussian(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="naive_bayes_gaussian",
            desc="Gaussian Naive Bayes classification for numerical features.",
            documentation=(
                "Fits a Gaussian Naive Bayes classifier for a nominal outcome "
                "using numerical features. Feature values are modeled with "
                "class-conditional Gaussian distributions, and missing values "
                "are not imputed by the classifier.\n\n"
                "The result includes class labels, class counts, class priors, "
                "per-class feature means, per-class feature variances, and "
                "feature names.\n\n"
                "Reference behavior is aligned with scikit-learn GaussianNB "
                "methodology, using aggregated class counts, means, and "
                "variances without sharing raw data."
            ),
            label="Gaussian Naive Bayes",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
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
                    desc="Numerical features used for classification.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                ),
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self) -> NaiveBayesGaussianResult:
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        labels = sorted(get_enum_codes(self.metadata, y_var))

        params = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
            },
            identical_results=True,
        )
        return NaiveBayesGaussianResult(**params)


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    labels,
):
    X = data[x_vars].to_numpy(dtype=float, copy=False)
    y = data[y_var].to_numpy()

    model = FederatedGaussianNB(
        x_vars=x_vars,
        labels=labels,
    )
    results = model.fit(X, y, agg_client=agg_client)

    return {
        "classes": list(results.labels),
        "class_count": results.class_count.astype(float).tolist(),
        "class_prior": results.class_prior.astype(float).tolist(),
        "theta": results.theta.tolist(),
        "var": results.var.tolist(),
        "feature_names": list(results.x_vars),
    }
