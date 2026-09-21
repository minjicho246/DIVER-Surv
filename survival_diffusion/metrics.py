"""Survival evaluation metrics."""

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def compute_hr(data, median_risk, min_count=5):
    """Return the high/low event-rate ratio, or NaN for insufficient data."""
    columns = ["risk_score", "event", "time"]
    if not set(columns).issubset(data.columns):
        return np.nan

    subset = data[columns].copy()
    for column in columns:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    subset = subset[(subset["time"] > 0) & subset["event"].isin([0, 1])]
    if subset.empty:
        return np.nan

    subset["group"] = (subset["risk_score"] >= median_risk).astype(int)
    aggregate = subset.groupby("group").agg(
        total_events=("event", "sum"),
        total_time=("time", "sum"),
        count=("event", "count"),
    )
    if set(aggregate.index) != {0, 1} or (aggregate["count"] < min_count).any():
        return np.nan

    low_rate = aggregate.loc[0, "total_events"] / aggregate.loc[0, "total_time"]
    high_rate = aggregate.loc[1, "total_events"] / aggregate.loc[1, "total_time"]
    if low_rate == 0:
        return np.nan
    return float(high_rate / low_rate)


def calc_metrics(times, risks, events, median_risk=None):
    """Compute C-index, log-rank p-value, event-rate ratio, and risk threshold."""
    times = np.asarray(times).reshape(-1)
    risks = np.asarray(risks).reshape(-1)
    events = np.asarray(events).reshape(-1)

    try:
        c_index = concordance_index(times, -risks, events)
    except Exception as error:
        print(f"C-index error: {error}")
        c_index = 0.0

    data = pd.DataFrame({"time": times, "risk_score": risks, "event": events})
    threshold = median_risk if median_risk is not None else np.median(risks)
    hr = compute_hr(data, threshold)
    p_value = -1.0
    low = data[data["risk_score"] < threshold]
    high = data[data["risk_score"] >= threshold]
    if not low.empty and not high.empty:
        try:
            p_value = logrank_test(low["time"], high["time"], low["event"], high["event"]).p_value
        except Exception:
            pass
    return c_index, p_value, hr, threshold
