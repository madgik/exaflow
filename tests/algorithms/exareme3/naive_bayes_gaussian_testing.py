from collections import Counter
from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.naive_bayes import FederatedGaussianNB
from exaflow.worker_communication import BadUserInput

ALGNAME_PRED = "test_nb_gaussian_predict"


def _sorted_labels(metadata: dict, y_var: str) -> List[str]:
    return sorted(metadata[y_var]["enumerations"].keys())


def _prepare_dataframe(data, x_vars: List[str], y_var: str):
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    return data[cols].copy()


class GaussianNBTestingPredict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: Dict[str, int]

    def run(self):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Gaussian NB predict requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        labels = _sorted_labels(self.metadata, y_var)

        udf_results = self.run_local_udf(
            func=gaussian_nb_predict_udf,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
            },
        )

        total = Counter()
        for worker_res in udf_results:
            total.update(worker_res["predictions"])

        return self.Result(predictions=dict(total))


@exareme3_udf(with_aggregation_server=True)
def gaussian_nb_predict_udf(
    agg_client,
    data,
    y_var,
    x_vars,
    labels,
):
    df = _prepare_dataframe(data, x_vars, y_var)

    X = df[x_vars].to_numpy(dtype=float, copy=False)
    y = df[y_var].to_numpy()

    model = FederatedGaussianNB(x_vars=x_vars, labels=labels)
    results = model.fit(X, y, agg_client=agg_client)

    if df.shape[0] == 0 or results.nobs == 0 or not results.labels:
        return {"predictions": {}}

    preds = results.predict(X)
    counts = Counter(preds.tolist())
    predictions = {str(label): int(count) for label, count in counts.items()}
    return {"predictions": predictions}
