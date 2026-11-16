# Metric and Concept Glossary

- **precision**: `tp / (tp + fp)` on the test rows at the stated threshold. Not the share of fraud among all transactions.
- **recall**: `tp / (tp + fn)`. Share of fraud rows flagged.
- **f1**: `2 * precision * recall / (precision + recall)`.
- **roc_auc**: Area under ROC curve on test rows. Can look high with extreme class imbalance.
- **average_precision**: scikit-learn `average_precision_score` on test rows (area under precision-recall curve).
- **prevalence**: `fraud rows / rows` in the test set. PaySim whole-file value is approximately 0.13%; earlier 8% test sets were class-balanced samples, not true operational prevalence.
- **false_alarms_per_10k**: `fp / rows * 10000`.
- **threshold**: Score cutoff chosen on the validation rows (maximising F1 at true prevalence), never tuned on the test rows.
- **fraud_value_caught**: Sum of `amount` over true alerts; **fraud_value_missed** = sum of `amount` over fraud rows not alerted. Values in PaySim simulated currency units.
- **net_saving**: `fraud_value_caught - (review_cost * alerts)` (interactive dashboard input). Note: break-even and cost model terms are NOT used in this project.
- **community_alert_rate (legacy)**: Share of discovered communities containing at least one flagged transaction. Not a ground-truth fraud ring detection rate.
- **ring recovered (planted test)**: `>= 80%` of the planted ring's accounts appear in one detected community AND `>= 80%` of that community's accounts belong to that ring.
- **homogeneity / completeness / v_measure**: scikit-learn clustering metrics computed on planted accounts only (true ring id versus detected community id).
- **speedup**: `time(local[1]) / time(local[n])` at the identical data fraction.
- **peak_rss**: Largest sum of resident memory over the Python process and its child processes (including Spark JVM worker/executor processes), sampled once per second.
