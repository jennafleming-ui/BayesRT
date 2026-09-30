"""Shared lap-time cleaning for BayesRT's training and serving code.

A fixed absolute window (the old 70s < LapTime < 95s filter) only matches one
circuit's pace. Applied across circuits it silently drops most laps from
slower tracks (e.g. Bahrain and Japan, ~97-98s average) while barely
touching faster ones (Monaco, Spain, ~80s average), which skews training data
and breaks any evaluation that holds out a slower circuit. Per-race IQR
fencing adapts the cutoff to each circuit's own lap-time distribution instead.
"""

import pandas as pd

DEFAULT_GROUP_COL = "Race"
DEFAULT_IQR_MULTIPLIER = 1.5  # standard Tukey fence


def clean_racing_laps(df: pd.DataFrame, group_col: str = DEFAULT_GROUP_COL,
                       k: float = DEFAULT_IQR_MULTIPLIER) -> pd.DataFrame:
    """Drop pit stops, incidents, and other non-racing laps.

    Keeps laps within [Q1 - k*IQR, Q3 + k*IQR] of their own group's
    (default: race's) lap-time distribution, so a single-circuit dataset and
    a multi-circuit one are cleaned consistently.
    """
    if group_col not in df.columns:
        group_col = None

    def _fence(group: pd.DataFrame) -> pd.DataFrame:
        q1, q3 = group["LapTimeSeconds"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return group[group["LapTimeSeconds"].between(lo, hi)]

    if group_col:
        return (df.groupby(group_col, group_keys=False)
                  .apply(_fence, include_groups=False)
                  .join(df[[group_col]], how="left"))
    return _fence(df)
