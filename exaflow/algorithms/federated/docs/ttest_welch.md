# Welch's T-Test

## Family

- `statistics`

## Purpose

`FederatedTtestWelch` computes an unequal-variance two-sample t-test from
aggregated sufficient statistics. Each worker contributes, per selected group:

- count
- sum
- sum of squares

The controller-facing result is computed from global means, sample variances,
Welch standard error, and Welch-Satterthwaite degrees of freedom.

## Notes

- Requires aggregation server support.
- Each selected group must have at least two finite observations.
- The test reports `groupA - groupB`; one-sided alternatives are interpreted in
  that direction.
- Cohen's d uses the square root of the average of the two sample variances.
