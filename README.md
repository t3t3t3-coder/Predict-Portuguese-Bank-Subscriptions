# Will a Client Subscribe to a Term Deposit?

A data analysis and lead-prioritization tool built for a bank's telemarketing campaign, using a dataset of 41,188 (`bank-additional-full.csv`) phone-marketing contacts to identify which clients are most likely to subscribe to a term deposit, so outreach can be prioritized toward the leads most worth calling.

**NOTE:** Support Vector Machine (SVC) models in this analysis were trained on a smaller sample of the data (`bank-additional.csv`, 4,119 rows) rather than the full dataset (41,187 rows), because SVM training on the full dataset exceeded available memory and crashed the notebook. SVC results are directionally useful but not directly comparable to the full-dataset models — see "Where the model struggles" below.

**📓 Full analysis notebook:** [Predict_Bank_Subscription.ipynb](Predict_Bank_Subscription.ipynb)

---

## Business Problem

A bank's call center can't call every client in its database. Agent time is limited, and most calls don't convert. Only about 11% of clients in this dataset subscribed. This project identifies which clients are most likely to say "yes" to a term deposit offer, so the bank can prioritize its calling list toward the leads most worth pursuing, rather than calling everyone or calling at random.

Because a **missed subscriber costs far more than a wasted call**, a missed subscriber is lost revenue with likely no second chance, while an unnecessary call costs only a few minutes of agent time. This analysis prioritizes **recall** (catching as many true subscribers as possible) over precision, even at the cost of some wasted calls.

## Summary of Findings

**Our best model by recall is, somewhat surprisingly, the simplest one: a class-weighted Logistic Regression using only six "essential" columns** (the outcome of the previous campaign plus five economic indicators). It catches about **70% of actual subscribers** (recall 0.697), trains in roughly 0.15 seconds, and needs almost none of the client-specific data the other models rely on. The cost is precision — only about 29% of the clients it flags as "likely yes" actually subscribe, so the majority of calls it recommends won't convert.

Three other models land close behind on recall, each with a different tradeoff:

| Model | Recall | Precision | Accuracy | Train Time |
|---|---|---|---|---|
| Logistic Regression (essential columns) | 0.697 | 0.290 | 0.773 | 0.15s |
| Logistic Regression (balanced, full features) | 0.648 | 0.368 | 0.835 | 0.98s |
| Decision Tree (tuned) | 0.617 | 0.387 | 0.847 | 0.22s |
| SVC Linear (balanced) | 0.589 | 0.358 | 0.840 | 11.20s |

Compared to a baseline of catching **zero** subscribers (always predicting "no"), all four of these represent a dramatic improvement on the metric that matters most for this business problem. Notably, the full-feature balanced Logistic Regression and the tuned Decision Tree both offer a better precision/accuracy balance than the essential-columns model while still capturing most of the recall — and the balanced linear SVM, which led every earlier version of this comparison, now trails all three while taking roughly **50–75x longer to train**, making it the weakest choice on both dimensions once the underlying data-cleaning issues were fixed.

Whether the essential-columns model's speed and recall are worth its precision cost or whether the more balanced tradeoff of the full-feature Logistic Regression or Decision Tree is preferable — depends on the dollar cost of a call versus the dollar value of a subscriber, a judgment call for the business rather than the model. All thirteen model/configuration results are in the full comparison table in the notebook.

### What actually drives subscription

A full coefficient/feature-importance breakdown is left to the notebook (no permutation importance or coefficient analysis has been performed on the final models yet — see Recommendations). Two things are clear from the modeling process so far:

- **The previous campaign's outcome and macroeconomic indicators appear to be the strongest signal in the dataset.** The essential-columns model, built from just `poutcome`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, and `nr.employed`, achieved the *highest* recall of all thirteen models/configurations tested — outperforming every model that had access to the full ~20-feature client profile. That's a strong signal that campaign timing and economic climate matter more to this outcome than most individual client attributes do.
- **Class imbalance is the central challenge, and rebalancing is what separates the top and bottom of the table.** With subscribers making up only 11% of the dataset, unbalanced models cluster at the bottom of the recall ranking.  SVC RBF, SVC Poly (tuned), SVC Linear, and base Logistic Regression all scored recall between 0.08 and 0.22, despite some of them having the *highest* precision and accuracy in the entire comparison. `class_weight='balanced'` (or, for the essential-columns model, a reduced feature set that happens to concentrate signal) was the deciding factor behind every model in the top half of the table.

### Where the model struggles

- **No single model wins on every metric — precision and recall pull in opposite directions across the table.** The four highest-recall models (essential-columns Logistic, balanced Logistic, tuned Decision Tree, balanced SVC) all sit at 29–39% precision; the highest-precision models (SVC RBF at 82% precision, SVC Linear at 77%) all have recall under 20%. There is no model in this comparison that is simultaneously strong on both — the choice of model is really a choice of where on that tradeoff the business wants.
- **SVM results aren't on equal footing with the other models, and the balanced SVM no longer justifies its cost.** Because the full dataset crashed during SVM training, all SVC variants were trained and evaluated on a 10% subsample rather than the full data. On top of that comparability gap, the balanced linear SVM now trains in **11.2 seconds versus roughly 0.1–1.0 seconds** for every other model in the table, while also placing fourth on recall. Unless further tuning changes this picture, SVMs are the weakest choice in this comparison on both speed and the primary business metric.
- **Hyperparameter tuning for SVMs is incomplete.** A `GridSearchCV` pass over SVM hyperparameters was attempted but repeatedly crashed the kernel.   The current SVM results reflect hand-picked settings, not a systematically tuned model, and it's possible tuning would improve their standing.
- **No formal validation set was held out for model selection.** All 13 models/configurations were compared against the same single test set. All models should be re-validate with fresh values.

## Recommendations

1. **Adopt a recall-first model for lead prioritization, choosing between three strong candidates rather than defaulting to SVM.** The essential-columns Logistic Regression offers the best recall (0.697) and is dramatically cheaper to train; the full-feature balanced Logistic Regression and tuned Decision Tree trade a bit of recall (0.65 and 0.62) for meaningfully better precision (0.37 and 0.39) and accuracy. Given how close these three are, the deciding factor should be the business's actual cost-per-call versus value-per-subscriber, not model complexity.
2. **Deprioritize the SVM approach unless further tuning changes its standing.** With the data-cleaning fixes in place, the balanced SVM no longer leads on recall and trains 10–100x slower than every other model — the memory/kernel-crash issues it caused are no longer clearly worth solving unless a completed hyperparameter search meaningfully improves its results.
3. **Hold out a dedicated validation set (or use k-fold cross-validation)** before finalizing a model for production, so the reported performance isn't inflated by having been used to compare 13 configurations against the same test data.
4. **Run a feature-importance analysis on the leading candidates**, particularly the essential-columns model if `poutcome` and the economic indicators are really driving most of the signal, that's a simpler and more explainable story for the bank than a full client-profile model, and worth confirming directly with coefficients.
5. **Treat the model as a prioritization tool, not an autopilot.** At 29–39% precision across the top four models, most recommended calls still won't convert — the model should rank and filter the calling list, with agents still exercising judgment on individual clients.
6. **Consider additional features** not in this dataset client tenure with the bank, existing product holdings, or channel/timing preferences — which likely explain some of the remaining prediction error, particularly for the lower-recall, higher-precision models that are currently missing most true subscribers.

## How to Use the Notebook

This analysis does not yet include a standalone prediction tool. To reproduce the results:

1. Open the notebook.
2. Run **Kernel → Restart & Run All** to execute every cell in order (this is required — several cells depend on state set up in earlier cells, and timing values are only meaningful on a clean run).
3. Scroll to the **Comparing Models** section near the end for the full accuracy/precision/recall/train-time comparison table and the accompanying charts.

## Methodology (Technical Summary)

- **Data:** 41,188 phone-marketing contact records (UCI Bank Marketing dataset) for the full-dataset models; a ~4,119-row subsample (`bank-additional.csv`) for SVM models, due to memory constraints on the full dataset
- **Target:** binary — did the client subscribe to a term deposit (`y`)
- **Key preprocessing decisions:**
  - Dropped `duration` (last call length) — a leakage feature, since it's only known after the call outcome is already determined
  - `default`, `housing`, and `loan` are one-hot encoded rather than collapsing `'unknown'` into `'no'`, preserving the distinction between a confirmed "no" and an undisclosed status
  - `pdays`'s sentinel value (`999`, meaning "never previously contacted") was split into a real numeric day-count plus a separate `not_previously_contacted` binary flag, rather than left as a distorting outlier value
- **Models tested:** Logistic Regression (default and class-weighted), K-Nearest Neighbors (default and tuned), Decision Tree (default and tuned, max depth 4), Support Vector Classifier (linear, linear balanced, polynomial, polynomial tuned, RBF, sigmoid), and a reduced-feature Logistic Regression using only previous-outcome and economic indicators — 13 models/configurations in total
- **Best model (by recall):** Logistic Regression, `class_weight='balanced'`, essential columns only (`poutcome` + 5 economic indicators) — Test Recall 0.697, Test Precision 0.290, Test Accuracy 0.773, Train Time 0.15s
- **Runner-up candidates:** Logistic Regression, balanced, full feature set (Recall 0.648, Precision 0.368, Accuracy 0.835, Train Time 0.98s); Decision Tree, tuned, max depth 4 (Recall 0.617, Precision 0.387, Accuracy 0.847, Train Time 0.22s)
- **Evaluation metric:** Recall was prioritized as the primary decision metric, driven by the business cost asymmetry between missed subscribers and wasted calls (Problem 4). Accuracy, precision, and training time (via `time.time()` around each model's `.fit()` call) were tracked alongside it for a fuller comparison.
- **Baseline:** always predicting the majority class ("no") yields 88% accuracy, but 0% recall and undefined precision — the model that "looks" most accurate is actually useless for the business goal.

---

*See the full notebook for the complete model comparison table (all ~13 configurations), confusion matrices, and detailed preprocessing steps.*
