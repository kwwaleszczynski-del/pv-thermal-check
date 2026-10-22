"""Temperature losses and performance ratio: raw PR (IEC 61724-1), PR corrected to 25 degC (IEC 61724-1 PR_STC)
and NREL's weather-corrected PR (Dierauf et al. 2013, NREL/TP-5200-57991), which corrects to the plant's own
irradiance-weighted annual mean cell temperature instead of 25 degC.

Usage:
    loss = temperature_loss(g, t_cell, gamma)                     # monthly energy-weighted loss [-]
    pr = monthly_pr(energy_kwh, g, t_cell, pdc0_kw, gamma, mask)  # columns pr_raw, pr_stc, pr_nrel
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cell_from_back(t_back: pd.Series, g: pd.Series, delta_t: float = 3.0) -> pd.Series:
    """SAPM conversion of a back-of-module reading to cell temperature (open rack: 3 K at 1000 W/m2)."""
    return t_back + g / 1000 * delta_t


def temperature_factor(t_cell, gamma: float, t_ref: float = 25.0):
    return 1 + gamma * (np.asarray(t_cell) - t_ref)


def temperature_loss(g: pd.Series, t_cell: pd.Series, gamma: float, freq: str | None = "MS") -> pd.Series | float:
    """Irradiance-weighted thermal loss relative to 25 degC: sum(G * gamma * (Tc - 25)) / sum(G). Negative = loss.
    freq=None returns the total over the series."""
    ok = g.notna() & t_cell.notna() & (g > 0)
    w = (g * gamma * (t_cell - 25))[ok]
    if freq is None:
        return float(w.sum() / g[ok].sum())
    return w.resample(freq).sum() / g[ok].resample(freq).sum()


def monthly_pr(energy: pd.Series, g: pd.Series, t_cell: pd.Series, pdc0: float, gamma: float, mask: pd.Series,
               t_typ: float | None = None, step_h: float = 0.25) -> pd.DataFrame:
    """Monthly PR on the rows in `mask`. energy is the power series [kW] (AC or DC), g the POA [W/m2].
    t_typ: the NREL reference cell temperature; by default the irradiance-weighted mean over the mask."""
    m = mask & energy.notna() & g.notna() & t_cell.notna()
    e, gg, tc = energy[m] * step_h, g[m], t_cell[m]
    if t_typ is None:
        t_typ = float((gg * tc).sum() / gg.sum())
    ref = pdc0 * gg / 1000 * step_h
    out = pd.DataFrame({
        "pr_raw": e.resample("MS").sum() / ref.resample("MS").sum(),
        "pr_stc": e.resample("MS").sum() / (ref * temperature_factor(tc, gamma)).resample("MS").sum(),
        "pr_nrel": e.resample("MS").sum() / (ref * temperature_factor(tc, gamma, t_typ)).resample("MS").sum(),
        "t_cell_w": (gg * tc).resample("MS").sum() / gg.resample("MS").sum(),
        "insolation_kwh_m2": (gg * step_h / 1000).resample("MS").sum(),
    })
    out.attrs["t_typ"] = t_typ
    return out


def seasonality(pr: pd.Series, t_cell_w: pd.Series) -> dict:
    """How seasonal a monthly PR series is: max-min spread and the slope against the monthly mean cell temperature."""
    ok = pr.notna() & t_cell_w.notna()
    slope = np.polyfit(t_cell_w[ok], pr[ok], 1)[0]
    return {"spread": float(pr[ok].max() - pr[ok].min()), "std": float(pr[ok].std()),
            "slope_per_k": float(slope), "corr_t": float(np.corrcoef(t_cell_w[ok], pr[ok])[0, 1])}
