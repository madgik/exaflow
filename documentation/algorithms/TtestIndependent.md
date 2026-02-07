<b><h2><center>Independent T-Test </center></h1></b>

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

<b><h4> Notation </h4></b>
Each local dataset *D<sup>(l)</sup>*, where *l*=1,...,*L*, is represented as a matrix of size *n* x *p*, where *L* is the number of medical centers, *n* is the number of points (patients) and *p* is the number of attributes. The elements of the above matrix can either be continuous or discrete (categorical).

In each local dataset, the independent attributes are denoted as a matrix *X<sup>(l)</sup>* and the dependent variable is denoted as a vector *y<sup>(l)</sup>*. *x*<sub>(*ij*)</sub><sup>(*l*)</sup> is the value of the *i*<sup>(*th*)</sup> patient of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital, while *x*<sub>(*j*)</sub><sup>(*l*)</sup> denotes the vector of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital. For categorical attributes, we use the notation *C*<sub>m</sub> <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> { *C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>M</sub>} for their domain.

<b><h4> Algorithm Description </h4></b>
The Student’s Independent samples t-test (sometimes called a two-samples t-test) is used to test the null hypothesis that two groups have the same mean. A low p-value suggests that the null hypothesis is not true, and therefore the group means are different. In each local dataset, let *x* and *y* be the variables of interest.*y* is the grouping variable with two levels.

![pseudo](images/independent_ttest_pseudocode.png)

<b><h4>Exareme3 Notes</h4></b>

- The grouping variable must have exactly two levels.
- Returns t-statistic, p-value, confidence interval and group statistics.

<b><h4>Algorithm Implementation</b></h4>

[Independent T-test](../../exaflow/algorithms/exareme3/ttest_independent.py)
