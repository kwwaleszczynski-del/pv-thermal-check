"""Oil-immersed step-up transformer under a PV load profile: the IEC 60076-7 (loading guide) thermal model and
insulation ageing.

Top-oil rise, and the hot-spot rise as the difference of two first-order terms (the winding's fast rise minus the
oil's slow catch-up), each driven by the per-unit load K:
    d(dTo)/dt  = ([(1 + R K^2) / (1 + R)]^x * dTor - dTo) / (k11 * tau_o)
    d(dTh1)/dt = (k21 * dThr * K^y - dTh1) / (k22 * tau_w)
    d(dTh2)/dt = ((k21 - 1) * dThr * K^y - dTh2) / (tau_o / k22)
    T_hotspot  = T_ambient + dTo + dTh1 - dTh2
Each equation is integrated exactly for a load held constant over the step (stable at any time step).

Relative ageing of non-thermally-upgraded paper: V = 2^((T_hs - 98) / 6), i.e. V = 1 at 98 degC.
Thermally upgraded paper: V = exp(15000 / 383 - 15000 / (T_hs + 273)).

The default parameters are the loading guide's exponents and time constants for ONAN distribution transformers,
with rated rises chosen so that the hot-spot is at the 98 degC reference at rated load and 20 degC ambient. They
describe a generic unit, not a specific transformer.

Usage:
    th = hotspot(k_pu, t_ambient, step_min=15)        # DataFrame: top_oil, hot_spot
    v = ageing_rate(th["hot_spot"])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pv_thermal.thermal import first_order_lag


@dataclass(frozen=True)
class TransformerParams:
    r: float = 5.0  # load losses / no-load losses at rated current
    x: float = 0.8  # oil exponent
    y: float = 1.6  # winding exponent
    k11: float = 1.0
    k21: float = 1.0
    k22: float = 2.0
    tau_o: float = 180.0  # oil time constant [min]
    tau_w: float = 4.0  # winding time constant [min]
    d_oil_rated: float = 52.0  # top-oil rise at rated load [K]
    d_hs_rated: float = 26.0  # hot-spot to top-oil gradient H*gr at rated load [K]


ONAN_DISTRIBUTION = TransformerParams()
LIMIT_HOT_SPOT = 120.0  # degC, distribution transformers, normal cyclic loading (IEC 60076-7)
LIMIT_TOP_OIL = 105.0


def hotspot(k: pd.Series, t_amb: pd.Series, p: TransformerParams = ONAN_DISTRIBUTION, step_min: float | None = None
            ) -> pd.DataFrame:
    """Top-oil and hot-spot temperature [degC] for a per-unit load series k and ambient t_amb on the same index."""
    if step_min is None:
        step_min = (k.index[1] - k.index[0]).total_seconds() / 60
    kk = k.fillna(0).clip(lower=0).to_numpy(float)
    u_o = ((1 + p.r * kk ** 2) / (1 + p.r)) ** p.x * p.d_oil_rated
    u_h1 = p.k21 * p.d_hs_rated * kk ** p.y
    u_h2 = (p.k21 - 1) * p.d_hs_rated * kk ** p.y
    d_o = first_order_lag(u_o, p.k11 * p.tau_o, step_min)
    d_h = first_order_lag(u_h1, p.k22 * p.tau_w, step_min) - first_order_lag(u_h2, p.tau_o / p.k22, step_min)
    ta = t_amb.to_numpy(float)
    return pd.DataFrame({"top_oil": ta + d_o, "hot_spot": ta + d_o + d_h}, index=k.index)


def ageing_rate(t_hs: pd.Series, upgraded: bool = False) -> pd.Series:
    if upgraded:
        return np.exp(15000 / 383 - 15000 / (t_hs + 273))
    return 2 ** ((t_hs - 98) / 6)


def sizing_table(p_ac: pd.Series, t_amb: pd.Series, p_inv_kw: float, ratios, pf: float = 1.0,
                 params: TransformerParams = ONAN_DISTRIBUTION) -> pd.DataFrame:
    """For each transformer rating ratio S_r / P_inverter: peak load, peak hot-spot and top-oil, hours above the
    98 degC reference and the 120 degC limit, and the ageing in days per year (365 = nominal rate)."""
    step_h = (p_ac.index[1] - p_ac.index[0]).total_seconds() / 3600
    years = len(p_ac) * step_h / 8760
    rows = []
    for r in ratios:
        k = p_ac.clip(lower=0) / (pf * r * p_inv_kw)
        th = hotspot(k, t_amb, params)
        v, vu = ageing_rate(th["hot_spot"]), ageing_rate(th["hot_spot"], upgraded=True)
        rows.append({"ratio": r, "k_max": float(k.max()), "hot_spot_max": float(th["hot_spot"].max()),
                     "top_oil_max": float(th["top_oil"].max()),
                     "h_above_98": float((th["hot_spot"] > 98).sum() * step_h / years),
                     "h_above_120": float((th["hot_spot"] > LIMIT_HOT_SPOT).sum() * step_h / years),
                     "ageing_days_per_year": float(v.sum() * step_h / 24 / years),
                     "ageing_days_per_year_upgraded": float(vu.sum() * step_h / 24 / years)})
    return pd.DataFrame(rows)


def smallest_ratio(table: pd.DataFrame) -> float:
    """Smallest rating ratio that keeps the hot-spot under 120 degC, top oil under 105 degC, the load under 1.5 p.u.
    and the ageing at or below the nominal rate."""
    ok = table[(table["hot_spot_max"] <= LIMIT_HOT_SPOT) & (table["top_oil_max"] <= LIMIT_TOP_OIL)
               & (table["k_max"] <= 1.5) & (table["ageing_days_per_year"] <= 365)]
    return float(ok["ratio"].min()) if len(ok) else float("nan")
