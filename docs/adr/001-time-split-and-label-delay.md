# 001: Split by time, and simulate label delay

## Context
Fraud data is ordered in time, and a fraud label arrives days or weeks after the transaction.
IEEE-CIS has transaction times but no chargeback dates.

## Decision
- Periods are cut on `TransactionDT`: train is the first 70% of transactions, valid the next 15%,
  holdout the last 15%. Transactions in the same second share a period.
- Each row gets a `label_available_at`. Fraud is confirmed after a lognormal delay (median 14
  days, clipped to 1 to 90). A transaction counts as legitimate after 30 days with no chargeback.
- A model trained at time `T` uses only rows with `label_available_at <= T`.
- The holdout period is loaded only by the live replay and the final evaluation.

## Alternatives
- **Random or stratified split:** this is the leak the project measures, so it is used only in
  the careless pipeline.
- **Ignore label delay:** measured at +0.027 validation ROC-AUC (0.981 against 0.954), so it is
  not a detail.

## Consequences
- The delay is independent of the transaction, which real chargebacks are not.
- In the 30 days before `T` only fraud rows are labelled. Dropping them changed live ROC-AUC by
  0.0004, so they are kept for training and excluded when choosing a threshold.
