"""Cleaning utilities for the Give Me Some Credit dataset.

The three delinquency-count columns encode 96 and 98 as placeholder values
rather than literal counts of late payments. Left as numbers they distort
every mean, scaler, and linear coefficient, so this module isolates the logic
for neutralizing them in one auditable place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The "magic numbers" and columns that drive the cleaning live here so an
# auditor can confirm what is being treated as a placeholder without reading
# the function body.
SENTINEL_VALUES = frozenset({96, 98})

LATE_PAYMENT_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

SENTINEL_FLAG_COLUMN = "late_sentinel"


def replace_late_payment_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 96/98 placeholder codes in the delinquency columns with NaN.

    Returns a new frame; the input is left untouched.
    """
    # Copy up front: callers may still need the raw frame to compare pre/post
    # distributions, and silently mutating an argument is a footgun.
    out = df.copy()

    # Fail loudly if the schema isn't what we assume, rather than surfacing an
    # opaque KeyError from deep inside pandas later on.
    missing = [c for c in LATE_PAYMENT_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Expected delinquency columns are absent: {missing}")

    n_rows = len(out)

    # Surface the counts before touching anything: the point is to let a human
    # confirm the 96/98 cluster matches the placeholder hypothesis instead of
    # trusting the replacement blindly.
    print("Sentinel occurrences per column (values 96/98):")
    for col in LATE_PAYMENT_COLUMNS:
        count = int(out[col].isin(SENTINEL_VALUES).sum())
        print(f"  {col}: {count}")

    # Capture the flag BEFORE replacing — once the values become NaN the
    # evidence is gone. That a row was coded this way may itself correlate with
    # default, so the signal is preserved rather than discarded with the value.
    out[SENTINEL_FLAG_COLUMN] = (
        out[LATE_PAYMENT_COLUMNS].isin(SENTINEL_VALUES).any(axis=1)
    )

    # NaN, not an imputed number: imputation estimates a parameter from the
    # data and must be fit inside CV folds (a sklearn Pipeline) to avoid
    # leakage. Baking one global value in here would leak across every split.
    # (Introducing NaN upcasts these columns to float, which is expected.)
    out[LATE_PAYMENT_COLUMNS] = out[LATE_PAYMENT_COLUMNS].mask(
        out[LATE_PAYMENT_COLUMNS].isin(SENTINEL_VALUES), other=np.nan
    )

    # Guard the two invariants a placeholder-replacement must hold: it neither
    # adds nor drops rows, and it actually removed every sentinel.
    assert len(out) == n_rows, "Row count changed; replacement must not add or drop rows."
    assert not out[LATE_PAYMENT_COLUMNS].isin(SENTINEL_VALUES).any().any(), (
        "Sentinel values (96/98) survived the replacement."
    )

    return out


# DebtRatio is debt divided by income, but when the income denominator is
# unusable (missing, $0, or the $1 placeholder) the field holds a raw dollar
# AMOUNT instead of a ratio — train-set evidence: median 0.29 when income is
# known vs ~1,162 when it isn't, and opposite default rates (12.6% for genuine
# ratios > 1 vs 5.3% for the artifact group). One column, two units.
UNUSABLE_INCOME_MAX = 1

DEBT_RATIO_FLAG_COLUMN = "debt_ratio_artifact"

# Credit contracts require legal adulthood, so a recorded age below 18 is a
# data error, not a young borrower.
MINIMUM_CREDIT_AGE = 18


def replace_debt_ratio_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """NaN out DebtRatio where income was unusable; keep a flag for the group.

    Returns a new frame; the input is left untouched.
    """
    out = df.copy()

    missing = [c for c in ("DebtRatio", "MonthlyIncome") if c not in out.columns]
    if missing:
        raise KeyError(f"Expected columns are absent: {missing}")

    n_rows = len(out)

    # With no real denominator the stored number is not a ratio, whatever its
    # magnitude — a "0.5" here means fifty cents of monthly debt, not 50%.
    # So the whole group goes to NaN, not just the visibly huge values.
    unusable = out["MonthlyIncome"].isna() | out["MonthlyIncome"].le(
        UNUSABLE_INCOME_MAX
    )

    # Surface what is about to happen, so the pattern stays verifiable rather
    # than trusted: most of the group should sit above 1 (dollar amounts).
    print("DebtRatio artifact rows (income missing/0/1):", int(unusable.sum()))
    print("  of which DebtRatio > 1:", int((unusable & out["DebtRatio"].gt(1)).sum()))
    print(
        "  genuine ratios > 1 kept:",
        int((~unusable & out["DebtRatio"].gt(1)).sum()),
    )

    # Same rationale as late_sentinel: membership in this group is itself
    # informative (it defaults BELOW base rate), so the signal is kept in a
    # flag while the misleading magnitude is removed.
    out[DEBT_RATIO_FLAG_COLUMN] = unusable
    out["DebtRatio"] = out["DebtRatio"].mask(unusable, other=np.nan)

    assert len(out) == n_rows, "Row count changed; repair must not add or drop rows."
    assert out.loc[out[DEBT_RATIO_FLAG_COLUMN], "DebtRatio"].isna().all(), (
        "Every flagged row must have DebtRatio set to NaN."
    )

    return out


def replace_impossible_ages(df: pd.DataFrame) -> pd.DataFrame:
    """NaN out ages below the legal contracting age (a lone age-0 row in train).

    Returns a new frame; the input is left untouched.
    """
    out = df.copy()
    n_rows = len(out)

    impossible = out["age"].lt(MINIMUM_CREDIT_AGE)
    print("Impossible ages (<18):", int(impossible.sum()))

    # NaN rather than a guessed value, for the usual reason: any filled number
    # is a parameter, and parameters get learned inside the CV folds.
    out["age"] = out["age"].mask(impossible, other=np.nan)

    assert len(out) == n_rows, "Row count changed; repair must not add or drop rows."
    assert not out["age"].lt(MINIMUM_CREDIT_AGE).any(), (
        "An age below the legal minimum survived the repair."
    )

    return out


def load_gmsc(path: str, training: bool = True) -> pd.DataFrame:
    """Load a Give Me Some Credit csv and apply every repair, in order.

    One entry point so train and Kaggle test cannot drift apart in cleaning.
    """
    # index_col=0 first: the unnamed ID column is unique per row, so leaving
    # it in defeats duplicate detection and would leak a row ID as a feature.
    df = pd.read_csv(path, index_col=0)

    if training:
        # Dedup is train-only: the Kaggle test file needs a prediction for
        # every Id, duplicated feature-rows included, and its index IS the Id
        # required by the submission format — so it is never reset either.
        df = df.drop_duplicates().reset_index(drop=True)

    df = replace_late_payment_sentinels(df)
    df = replace_debt_ratio_artifacts(df)
    df = replace_impossible_ages(df)
    return df
