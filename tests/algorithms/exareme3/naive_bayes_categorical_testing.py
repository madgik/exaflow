from collections import Counter
from typing import Dict

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes import FederatedCategoricalNB
from exaflow.algorithms.federated.preprocessing.ordinal_encoder import (
    FederatedOrdinalEncoder,
)
from exaflow.worker_communication import BadUserInput

ALGNAME_PRED = "test_nb_categorical_predict"


class Result(BaseModel):
    predictions: Dict[str, int]


class CategoricalNBTestingPredict(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name=ALGNAME_PRED,
            desc="Uses Bayes' theorem to calculate the probability of each class given a set of nominal features assuming independence between features. It then classifies data points based on the class with the highest probability.",
            documentation="Uses Bayes' theorem to calculate the probability of each class given a set of nominal features assuming independence between features. It then classifies data points based on the class with the highest probability.",
            label="Categorical Naive Bayes classifier with cross-validation",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable (dependent)",
                    desc="A unique nominal variable.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates (independent)",
                    desc="One or more nominal variables.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                ),
                validation=None,
            ),
            parameters={},
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Naive Bayes categorical predict requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        categories = {
            var: sorted(get_enum_codes(self.metadata, var)) for var in x_vars + [y_var]
        }

        udf_results = self.run_local_udf(
            func=categorical_nb_predict_udf,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
            },
        )

        total = Counter()
        for worker_res in udf_results:
            total.update(worker_res["predictions"])

        return Result(predictions=dict(total))


def _prepare_dataframe(data, x_vars, y_var):
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    df = data[cols].copy()

    return df


@exareme3_udf(with_aggregation_server=True)
def categorical_nb_predict_udf(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
):
    df = _prepare_dataframe(data, x_vars, y_var)

    encoder = FederatedOrdinalEncoder(
        categories=categories,
        handle_unknown="error",
    )
    encoder.fit(
        agg_client=agg_client,
        data=df,
        categorical_vars=x_vars,
    )
    X_enc = encoder.transform(
        df,
        categorical_vars=x_vars,
        numerical_vars=[],
    )
    y = df[y_var].to_numpy()

    model = FederatedCategoricalNB(y_var=y_var, x_vars=x_vars, categories=categories)
    results = model.fit(X_enc, y, agg_client=agg_client)

    if df.shape[0] == 0 or results.class_count.sum() == 0:
        return {"predictions": {}}

    preds = results.predict(X_enc)
    counts = Counter(preds.tolist())
    predictions = {str(label): int(count) for label, count in counts.items()}
    return {"predictions": predictions}
