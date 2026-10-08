# The story

## The claim

A fraud model that scores 0.999 offline can be a 0.90 model in production, and the difference is
made by ordinary habits, not exotic bugs. I wanted to measure that gap end to end instead of
asserting it.

## What I built

1. **A careless pipeline, on purpose.** Random split, card aggregates over the whole dataset, and
   a card fraud rate computed with every label. Each habit is added separately, so the score
   inflation can be attributed: 0.910 → 0.970 → 0.974 → 0.999 ROC-AUC.
2. **A point-in-time pipeline.** Every feature is defined once in a spec. A transaction's features
   use only the card's earlier transactions and only labels that had arrived by then. The model
   trains only on rows whose label had arrived by the training date.
3. **A live path.** Transactions and label arrivals are replayed through Redpanda. A consumer
   keeps per-card state in Redis and computes the same spec. A FastAPI service reads that state,
   scores, and then records the transaction.
4. **A measurement.** Both final models are scored on the holdout month using the features the
   stream produced.

## What I found

- **The careless model lost almost half its PR-AUC** (0.986 → 0.513). The honest model's estimate
  held (0.695 → 0.665).
- **The honest model is also the better model live** (0.947 against 0.901 ROC-AUC). Leakage does
  not only flatter the score, it produces a model that leans on features it will never get.
- **In money, the careless model promised $264 per 1,000 transactions and cost $4,304.** Its
  threshold was tuned on scores it could not reproduce.
- **Choosing the threshold by cost halved the cost** of the accuracy-chosen threshold ($147k
  against $300k for the month), and landed within 1.3% of the best threshold in hindsight.

## Things that were harder than expected

- **Label timing is its own leak.** Point-in-time features with labels assumed instantly known
  scored 0.981 on validation. With a realistic delay it was 0.954. Nothing about the features'
  timestamps was wrong; the labels were simply from the future.
- **Parity needs an ordering rule, not just the same formulas.** A label arriving in the same
  second as a transaction has to be applied first in both paths, and a transaction must not see
  itself. With transactions and labels on one ordered stream, the online features matched the
  offline ones on all 88,581 holdout rows.
- **The first scoring service took 121 ms per request.** Profiling showed Redis at 1.8 ms and the
  model under 1 ms. The rest was building a one-row DataFrame. A plain-array scorer brought it to
  2.8 ms with identical scores.
- **Recent training rows are all fraud.** A legitimate label takes 30 days, so in the last 30 days
  before training only fraud is labelled. I measured the effect of dropping those rows (live
  ROC-AUC 0.9465 against 0.9469) and kept them, but chose the threshold only on rows old enough
  to have both classes.

## What I would do next

- Repeat over several seeds and training dates, and report intervals.
- Simulate label delay that depends on the transaction, so the early monitoring view is biased
  the way it is in practice.
- Trim per-card history in Redis to the largest window and keep running sums for the rest.
- Partition the stream by card and measure latency under concurrent load.
