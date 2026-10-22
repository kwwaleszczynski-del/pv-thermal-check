"""Sensor fault detection from the residual of a module temperature model, validated by injecting faults into the
real sensor record.

Two daily features over the operating, lit part of each day (POA > 200 W/m2):
    resid = mean(T_measured - T_model)                  an offset or a drift moves it
    ratio = sum(T_measured - Ta) / sum(T_model - Ta)    a sensor losing thermal contact (< 1) or a change in rear
                                                        ventilation (> 1) moves it
Two-sided CUSUM charts on the standardised features raise the alarm; their threshold is set on a fault-free
reference year to at most one false alarm per year. A plain 3-sigma chart on the daily residual is the baseline.
After an alarm, the daily regression rise_measured = offset + gain * rise_model labels the fault.

Usage:
    feats = daily_features(t_mod, t_model, ta, mask, tz)
    ref = reference_stats(feats_reference_year)
    h = calibrate_h(feats_reference_year, ref)
    alarms = cusum_alarms(feats, ref, h)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FAULTS = ("offset_plus", "offset_minus", "drift", "detached", "ventilation")


def inject(t_mod: pd.Series, ta: pd.Series, g: pd.Series, kind: str, start: pd.Timestamp, size: float | None = None
           ) -> pd.Series:
    """Return a copy of the sensor series with a fault from `start` on.
    offset_plus/minus: step of +-size K (default 3); drift: ramp of size K per 90 days (default 4.5);
    detached: daytime rise over air scaled by size (default 0.6); ventilation: daytime rise scaled by size (1.3)."""
    out = t_mod.copy()
    on = out.index >= start
    if kind == "offset_plus":
        out[on] += 3.0 if size is None else size
    elif kind == "offset_minus":
        out[on] -= 3.0 if size is None else size
    elif kind == "drift":
        days = (out.index[on] - start).total_seconds() / 86400
        out[on] += (4.5 if size is None else size) * days / 90
    elif kind in ("detached", "ventilation"):
        f = (0.6 if kind == "detached" else 1.3) if size is None else size
        lit = on & (g > 20).to_numpy()
        out[lit] = ta[lit] + f * (out[lit] - ta[lit])
    else:
        raise ValueError(kind)
    return out


def daily_features(t_mod: pd.Series, t_model: pd.Series, ta: pd.Series, mask: pd.Series, tz: str,
                   min_points: int = 8) -> pd.DataFrame:
    """Per local day: mean residual, rise ratio, and the offset/gain regression used to label a fault."""
    rm, rp = (t_mod - ta)[mask], (t_model - ta)[mask]
    ok = rm.notna() & rp.notna()
    x, y = rp[ok], rm[ok]
    day = x.index.tz_convert(tz).normalize().tz_localize(None)
    s = pd.DataFrame({"n": 1.0, "x": x, "y": y, "xx": x * x, "xy": x * y}).groupby(day).sum()
    s = s[(s["n"] >= min_points) & (s["x"] > 0)]
    n = s["n"]
    sxx = s["xx"] - s["x"] ** 2 / n  # n * var(x)
    gain = ((s["xy"] - s["x"] * s["y"] / n) / sxx).where(sxx / n > 1.0)  # least squares, needs some spread in x
    out = pd.DataFrame({"resid": (s["y"] - s["x"]) / n, "ratio": s["y"] / s["x"],
                        "offset": (s["y"] - gain * s["x"]) / n, "gain": gain, "n": n.astype(int)})
    out.index.name = None
    return out.sort_index()


@dataclass
class Reference:
    mean: pd.Series
    std: pd.Series


def reference_stats(feats: pd.DataFrame) -> Reference:
    """Robust centre and spread (median, 1.4826 * MAD) of the daily features over a fault-free period."""
    cols = ["resid", "ratio", "offset", "gain"]
    med = feats[cols].median()
    mad = (feats[cols] - med).abs().median() * 1.4826
    return Reference(med, mad)


def cusum_alarms(feats: pd.DataFrame, ref: Reference, h: float = 6.0, k: float = 0.5,
                 cols: tuple[str, ...] = ("resid", "ratio")) -> pd.DataFrame:
    """Two-sided CUSUM per feature on z = (x - mean) / std, clipped at +-4 sigma so one odd day cannot alarm
    alone. Statistics restart from zero after an alarm. Returns one row per alarm (day, feature, direction)."""
    alarms = []
    for c in cols:
        z = ((feats[c] - ref.mean[c]) / ref.std[c]).clip(-4, 4)
        hi = lo = 0.0
        for day, v in z.items():
            hi, lo = max(0.0, hi + v - k), max(0.0, lo - v - k)
            if hi > h or lo > h:
                alarms.append((day, c, "up" if hi > h else "down"))
                hi = lo = 0.0
    return pd.DataFrame(alarms, columns=["day", "feature", "direction"]).sort_values("day", ignore_index=True)


def calibrate_h(feats: pd.DataFrame, ref: Reference, max_per_year: float = 1.0,
                grid: tuple[float, ...] = tuple(np.arange(3.0, 30.5, 0.5)),
                cols: tuple[str, ...] = ("resid", "ratio")) -> float:
    """Smallest CUSUM threshold with at most max_per_year alarms on the (fault-free) reference features."""
    years = len(feats) / 365.25
    for h in grid:
        if len(cusum_alarms(feats, ref, h, cols=cols)) <= max_per_year * max(years, 1.0):
            return float(h)
    return float(grid[-1])


def label(feats: pd.DataFrame, ref: Reference, day: pd.Timestamp, window: int = 14) -> str:
    """Name the likely fault from the `window` days after an alarm: a gain change means a contact or ventilation
    problem, otherwise an offset or drift."""
    x = feats[(feats.index >= day) & (feats.index < day + pd.Timedelta(days=window))]
    zg = (x["gain"].median() - ref.mean["gain"]) / ref.std["gain"]
    zo = (x["offset"].median() - ref.mean["offset"]) / ref.std["offset"]
    zr = (x["resid"].median() - ref.mean["resid"]) / ref.std["resid"]
    if abs(zg) >= 1.0 and abs(zg) >= abs(zo):
        return "loss of contact" if zg < 0 else "reduced ventilation"
    return "offset / drift up" if zr > 0 else "offset / drift down"


def shewhart_alarms(feats: pd.DataFrame, ref: Reference, limit: float = 3.0) -> pd.DataFrame:
    """Baseline: alarm on any day whose mean residual is beyond +-limit robust sigmas."""
    z = (feats["resid"] - ref.mean["resid"]) / ref.std["resid"]
    hit = z[z.abs() > limit]
    return pd.DataFrame({"day": hit.index, "feature": "resid", "direction": np.where(hit > 0, "up", "down")})


def first_alarm_after(alarms: pd.DataFrame, start: pd.Timestamp, horizon_days: int) -> float:
    """Days from fault start to the first alarm within the horizon, NaN if none."""
    if alarms.empty:
        return np.nan
    a = alarms["day"][(alarms["day"] >= start) & (alarms["day"] < start + pd.Timedelta(days=horizon_days))]
    return float((a.iloc[0] - start).days) if len(a) else np.nan
