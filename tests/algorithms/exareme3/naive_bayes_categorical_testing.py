from collections import Counter
from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes_categorical import FederatedCategoricalNB
from exaflow.algorithms.federated.transformers.ordinal_encoder import (
    FederatedOrdinalEncoder,
)
from exaflow.worker_communication import BadUserInput

ALGNAME_PRED = "test_nb_categorical_predict"


def _sorted_categories(metadata: dict, variables: List[str]) -> Dict[str, List[str]]:
    return {
        var: list(sorted(metadata[var]["enumerations"].keys())) for var in variables
    }


class CategoricalNBTestingPredict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: Dict[str, int]

    def run(self):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Naive Bayes categorical predict requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        categories = _sorted_categories(self.metadata, x_vars + [y_var])

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

        return self.Result(predictions=dict(total))


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
