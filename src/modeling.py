"""Baseline modeling for the Give Me Some Credit dataset.

The notebook orchestrates; this module owns the reusable logic. The test
split produced by split_data() is sealed — a test set consulted during
development stops measuring generalization and starts measuring how many
times you peeked. Its one sanctioned use is evaluate_holdout(), called once,
after model selection has finished on CV.

The progression lives in the functions: build_pipeline() is the linear
baseline the project was measured against; build_gbm_pipeline() is the
escalation that won model selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "SeriousDlqin2yrs"


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate features from target, return (X_train, X_test, y_train, y_test)."""
    y = df[target]
    X = df.drop(columns=[target])

    # stratify=y preserves the ~7% delinquency rate in both splits. With a
    # class this rare, an unstratified split can hand the two sides different
    # base rates, which shifts scores for reasons that have nothing to do
    # with the model. The test split returned here is sealed: nothing in this
    # module fits on it or scores on it — it gets spent exactly once, at
    # final evaluation.
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_pipeline() -> Pipeline:
    """Baseline pipeline: median-impute -> scale -> weighted logistic regression."""
    return Pipeline(
        [
            # The imputer LEARNS its fill values (per-column medians) from the
            # data it is fit on — which is exactly why it must live inside the
            # pipeline: cross-validation then re-fits it on each fold's
            # training portion, so no validation rows leak backward into the
            # fill values. Median rather than mean because MonthlyIncome and
            # DebtRatio have tails extreme enough to drag a mean far above a
            # typical borrower.
            #
            # add_indicator=True appends an is-missing flag per imputed
            # column: "income not reported" may itself carry risk signal, and
            # the model should get to weigh that fact rather than have the
            # gap silently papered over with a plausible number.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            # The scaler also LEARNS parameters (per-column mean/std) — same
            # leakage argument as the imputer. Beyond leakage hygiene, these
            # features span 0-1 ratios to seven-figure incomes; without a
            # shared scale, regularization and the solver both overweight
            # whichever columns happen to have the biggest units.
            ("scale", StandardScaler()),
            # class_weight="balanced" reweights the loss so the ~7% minority
            # matters as much as the 93% majority — otherwise the cheapest
            # route to a low loss is predicting "no default" for everyone.
            # max_iter raised because lbfgs at its default budget can stop
            # short on a dataset this size and leave a ConvergenceWarning.
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def evaluate_cv(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """5-fold stratified CV on the training split; prints and returns AUC scores.

    Returns the per-fold AUCs (mean and std are printed and derivable from
    them), so the notebook can plot or compare without re-running.
    """
    # Stratified folds for the same reason the outer split is stratified: at
    # a 7% positive rate, unstratified folds drift in base rate and that
    # drift shows up as noise in the fold scores.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Passing the WHOLE pipeline — not a pre-transformed X — is what makes
    # the per-fold re-fitting happen. Impute or scale once up front and every
    # fold's medians and means have already seen that fold's validation rows:
    # quiet leakage, optimistic score.
    scores = cross_val_score(
        pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1
    )

    # The std is part of the result, not decoration: 0.75 ± 0.002 and
    # 0.75 ± 0.03 are different objects. Any later "improvement" smaller
    # than this spread is noise, not progress.
    print(f"per-fold AUC: {np.round(scores, 4).tolist()}")
    print(f"CV AUC      : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def build_gbm_pipeline() -> Pipeline:
    """Final-model candidate: histogram gradient boosting on raw repaired data."""
    return Pipeline(
        [
            # No imputer, no scaler. The estimator routes NaN natively — each
            # split learns which branch missing values take — which suits this
            # dataset, where every NaN is a deliberate "unknown" marker (income
            # not reported, sentinel-coded delinquency, artifact DebtRatio).
            # Imputing first would erase exactly that routing information; and
            # trees split on thresholds, so scale is irrelevant.
            #
            # Config chosen by a 5-fold CV sweep on the training split
            # (defaults 0.8638 -> this 0.8649 ± 0.0019). The leaders sat
            # within about one fold-std of each other, so the pick favors the
            # most regularized of them: slower learning rate with more, and
            # smaller, trees. class_weight is omitted because AUC is a
            # ranking metric — reweighting shifts the decision threshold, not
            # the ordering (measured: 0.8639 balanced vs 0.8649 without).
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=600,
                    max_leaf_nodes=15,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_holdout(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_reference: np.ndarray | None = None,
) -> float:
    """Score an already-fitted model on the sealed test split. Call once.

    This is the single sanctioned point of contact with the test set in the
    whole codebase. It exists as a named function so that "touching the test
    set" is greppable rather than something that can happen quietly inline.
    """
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"HELD-OUT TEST AUC: {auc:.4f}   (this number is now spent)")

    # The point of the comparison: if the test score lands within ~2 fold-stds
    # of the CV mean, the CV protocol was honest and can keep being trusted;
    # a big gap would mean leakage or overfitting to the folds. Either way,
    # nothing gets tuned after this — reacting to the test number would turn
    # it into one more validation set.
    if cv_reference is not None:
        mean, std = cv_reference.mean(), cv_reference.std()
        print(
            f"CV estimate was  : {mean:.4f} ± {std:.4f}"
            f"   (gap {auc - mean:+.4f} = {(auc - mean) / std:+.1f} fold-stds)"
        )
    return auc
