# Give Me Some Credit — Credit Default Prediction

Predicting the probability that a borrower experiences serious delinquency (90+ days past due)
within two years, on the Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
dataset — 150,000 borrowers, 10 features, a 6.7% positive rate.

**Final model: histogram gradient boosting — held-out test AUC 0.8642** (5-fold CV 0.8649 ± 0.0019).
For reference, the competition's winning private-leaderboard score was ~0.8696.

The emphasis of this project is not the score. It is **finding the data's hidden encoding artifacts**
and **a methodology that is verifiably free of leakage** — the final test score landing within
0.4 fold-standard-deviations of the cross-validated estimate is the receipt for that.

---

## Results

| Stage | 5-fold CV AUC | What moved it |
|---|---|---|
| Logistic regression baseline | 0.8203 ± 0.0010 | a trustworthy floor first |
| + data repairs | 0.8208 ± 0.0012 | fixing *meaning* ≠ moving the *score* |
| + `log1p` on heavy-tailed features | 0.8562 ± 0.0020 | the tails were the linear model's binding constraint |
| **HistGradientBoosting** (final) | **0.8649 ± 0.0019** | native NaN routing; scale-invariant |
| **Sealed test set — evaluated once** | **0.8642** | gap −0.4 fold-std ⇒ CV was honest |

Every score is reported as mean ± standard deviation across folds. Any "improvement" smaller than
that spread is treated as noise, not progress.

## Three encoding artifacts found in the data

Each was found the same way: spot a discontinuity in a distribution, form a hypothesis, verify it
against outcome data, then repair it **without discarding the information**.

**1. The row-ID trap.** The csv's unnamed first column is a unique row identifier. Loaded as a
feature it makes every row look distinct, so `drop_duplicates()` silently removes **0** of the 609
real duplicates — and the ID leaks in as a predictor. Fixed with `index_col=0`.

**2. Sentinel codes 96 / 98.** The three delinquency-count columns taper smoothly to ~15, then
show a gap, then a cluster at 96 and 98 — 269 rows, co-occurring across all three columns
simultaneously, which marks a record-level "data unavailable" state rather than three independent
errors. Replaced with `NaN` plus a `late_sentinel` flag, because *being coded that way is itself
risk information*.

**3. `DebtRatio` holds two different units in one column.** It should be debt ÷ income (median
**0.29** where income is known), but where income is missing, `$0`, or `$1` — 31,433 rows — it
holds a raw dollar amount (median **~1,162**). Magnitude alone is suggestive; the decisive evidence
is risk:

| Group | n | Default rate |
|---|---|---|
| `DebtRatio` ≤ 1 | 91,436 | 6.8% |
| `DebtRatio` > 1, **real** income | 4,054 | **12.6%** |
| `DebtRatio` > 1, income missing/0/1 | 24,022 | **5.3%** |

The two groups point in **opposite directions**, so a single coefficient on the raw column averages
away a genuine 2× risk signal. Repaired to `NaN` + a `debt_ratio_artifact` flag.

*(Also: one row with `age == 0`, below the legal contracting age → `NaN`.)*

## Methodology

- **The test set is sealed.** Stratified 80/20 split, spent on exactly one final evaluation after
  model selection was complete. Contact with it happens only inside a single named function
  (`evaluate_holdout`), so "touching the test set" is greppable rather than something that can
  happen quietly inline.
- **Anything that learns a parameter lives inside the `Pipeline`** — imputer medians, scaler
  mean/std — so cross-validation re-fits them on each fold's training portion and no validation
  data leaks backward into the fill values. Parameter-free transforms (`log1p`) are applied
  outside the folds, because only *fitted* parameters can leak.
- **Missing values stay `NaN`.** Every `NaN` here is a deliberate "unknown" marker. Hand-filling
  them was tested and rejected: filling missing income with `$1` or `$0` scores 0.8200 vs 0.8203
  (inside the noise band) and misplaces a below-base-rate group next to the *highest*-risk income
  band — real income of \$1–2k defaults at 9.0%, while unknown income defaults at 5.5%.
- **The gradient booster gets no imputer and no scaler.** It routes `NaN` natively, turning
  missingness into a learned split direction, which is exactly right for data whose gaps are
  meaningful. Tree splits are threshold-based, so scaling and monotone transforms are irrelevant
  (verified: identical AUC on raw vs logged features).

## Repository layout

```
├── Give_me_some_credit.ipynb   # orchestration + narrative; runs top-to-bottom
├── src/
│   ├── cleaning.py             # sentinel/DebtRatio/age repairs + load_gmsc() entry point
│   ├── features.py             # log1p re-expression of heavy-tailed columns
│   └── modeling.py             # split, pipelines, cross-validation, held-out evaluation
├── PROJECT_SUMMARY.md          # condensed write-up of findings and decisions
└── submission.csv              # 101,503 Kaggle predictions (generated)
```

The notebook is the orchestration layer only — reusable logic lives in `src/`, commented to explain
*why* rather than *what*, with module-level constants for auditability and asserts guarding
invariants (row counts preserved, no sentinel values surviving). A single loader, `load_gmsc()`,
applies every repair to both the training and Kaggle test files so their cleaning cannot drift apart.

## Running it

Requires Python 3.13 with `pandas`, `numpy`, `scikit-learn` ≥ 1.7, `matplotlib`.

```bash
# place cs-training.csv and cs-test.csv in the project root (see Data below)
jupyter notebook Give_me_some_credit.ipynb   # Kernel -> Restart & Run All
```

Full run takes well under a minute. It produces the table above and writes `submission.csv`.

## Data

The dataset is distributed by Kaggle and is **not included in this repository** — download
`cs-training.csv` and `cs-test.csv` from the
[competition data page](https://www.kaggle.com/c/GiveMeSomeCredit/data) (a Kaggle account and
acceptance of the competition rules are required) and place them in the project root.

## Possible next steps

Feature interactions and model ensembling — judged by cross-validation only, since the held-out
test set has been spent and can no longer referee anything. Probability calibration would matter if
these scores fed real lending decisions, and monotonic constraints on the booster would help where
regulatory interpretability is required.

