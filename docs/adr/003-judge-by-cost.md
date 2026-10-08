# 003: Choose and judge the alert threshold by cost

## Context
At a 3.5% fraud rate, accuracy is maximised by alerting on very little. The business cares about
money lost to fraud and the cost of reviewing alerts.

## Decision
- `cost(threshold) = amount of fraud below the threshold + review_cost * alerts`.
  The review cost defaults to $5 and is configurable. A reviewed fraud is assumed to be stopped.
- The threshold is chosen before deployment, on rows the model held back for early stopping that
  are at least 30 days old (so both classes are labelled), then judged on live holdout scores.
- Each policy reports the cost it promised on its selection data next to the cost it delivered,
  and the best threshold in hindsight is shown as a bound.

## Alternatives
- **Per-transaction rule (alert when `p * amount > review_cost`):** it needs calibrated
  probabilities, which training on matured labels does not guarantee.
- **F1 or a fixed alert budget:** neither is in the unit the decision is made in.

## Consequences
- On the holdout period: $147,196 with the cost-chosen threshold, $299,514 with the
  accuracy-chosen one, $145,328 in hindsight.
- The figures move with the review cost, and they leave out customer friction from false alarms.
