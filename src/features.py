"""Feature engineering for the Give Me Some Credit dataset.

Only parameter-free transforms live here. Anything that LEARNS from data
(imputation medians, scaler statistics) belongs inside the sklearn Pipeline
so it re-fits per CV fold — a fixed mathematical re-expression like log1p
learns nothing, so applying it once, outside the folds, cannot leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# These columns span three to six orders of magnitude with long right tails.
# A linear model fits one coefficient per column, so a single extreme value
# (utilization of 50,708 against a median of 0.15) dominates the fit; log1p
# compresses the tail while preserving order. Tree models split on
# thresholds and are indifferent to monotone re-expression — this exists for
# the linear track.
HEAVY_TAIL_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace heavy-tailed columns with their log1p re-expression.

    Returns a new frame; the input is left untouched. NaN passes through
    (log1p(NaN) = NaN), so missingness survives for the pipeline to handle.
    """
    out = df.copy()

    # log1p demands non-negative input; a negative here would silently become
    # NaN and masquerade as missingness, so fail loudly instead.
    vals = out[HEAVY_TAIL_COLUMNS]
    assert ((vals >= 0) | vals.isna()).all().all(), (
        "Negative value in a heavy-tail column; log1p would corrupt it."
    )

    # Replace in place rather than adding *_log twins: the transform is a
    # re-expression of the same information, and keeping both would just
    # hand the model two collinear copies.
    out[HEAVY_TAIL_COLUMNS] = np.log1p(vals)
    return out
