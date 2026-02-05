<b><h2><center>Support Vector Machine (SVM)</center></h1></b>

<b><h4> Some General Remarks </h4></b>
The general architecture of the MIP follows a Master/Worker paradigm where many Workers, operating in multiple medical centers, are coordinated by one Master. Only Workers are allowed access to the anonymized data in each medical center and the Master only sees aggregate data, derived from the full data and sent to him by the Workers.

In this implementation, the general architecture of the MIP is followed to obtain a local model from each Worker. The global model derives from averaging all the local models, with the reliable and popular averaging technique, Federated Averaging (FedAvg).

Our naming convention is that procedures run on Workers are given the adjective _local_ whereas those running on Master are called _global_.

<b><h4>Algorithm Description</b></h4>

The SVM algorithm uses the state-of-art python library, scikit-learn to calculate the local models. The model from each Worker is then averaged on the Master to return the result of the averaging process.

<b><h4>Algorithm Implementation</b></h4>

[SVM](../../exaflow/algorithms/exareme3/linear_svm.py)

[Federated Averaging Strategy](../../exaflow/algorithms/exareme3/fedaverage.py)
