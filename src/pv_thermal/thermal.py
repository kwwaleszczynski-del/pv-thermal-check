"""Module temperature models: steady-state Faiman, SAPM, PVsyst and Ross (all through pvlib), their published
default coefficients, least-squares fitting, and two ways to add thermal inertia (a fitted first-order lag and the
Prilliman et al. 2020 transient model).

Inputs everywhere are POA irradiance g [W/m2], air temperature ta [degC] and wind speed ws [m/s, 10 m height].

Usage:
    fit = fit_model("faiman", g, ta, ws, target)          # -> FitResult(params=..., rmse=...)
    pred = predict("faiman", fit.params, g, ta, ws)
    dyn = fit_model("faiman", g, ta, ws, target, dynamic=True)   # adds a time constant tau [min]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import least_squares
from scipy.signal import lfilter

log = logging.getLogger(__name__)

ALPHA = 0.9  # PVsyst absorption coefficient (pvlib default)

# Published defaults. Faiman: IEC 61853-2 / pvlib; SAPM: open rack, glass/polymer back-sheet (King et al. 2004);
# PVsyst: free-standing array; Ross: k = (NOCT - 20) / 800 with the module's datasheet NOCT of 45.4 degC.
DEFAULTS = {
    "faiman": {"u0": 25.0, "u1": 6.84},
    "sapm": {"a": -3.56, "b": -0.075},
    "pvsyst": {"u_c": 29.0, "u_v": 0.0},
    "ross": {"k": (45.4 - 20) / 800},
}
BOUNDS = {
    "faiman": ([1.0, 0.0], [300.0, 100.0]),
    "sapm": ([-8.0, -1.0], [0.0, 0.0]),
    "pvsyst": ([1.0, 0.0], [300.0, 100.0]),
    "ross": ([0.0], [0.2]),
}
SAPM_DT = 3.0  # back-of-module to cell difference at 1000 W/m2, open rack


def steady(model: str, p: dict, g, ta, ws, efficiency: float = 0.0, cell: bool = False):
    """Steady-state temperature. `efficiency` is the electrical efficiency for PVsyst (0 at open circuit).
    For SAPM, cell=False gives the back-of-module temperature and cell=True adds the 3 K cell offset."""
    if model == "faiman":
        return pvlib.temperature.faiman(g, ta, ws, u0=p["u0"], u1=p["u1"])
    if model == "sapm":
        if cell:
            return pvlib.temperature.sapm_cell(g, ta, ws, p["a"], p["b"], SAPM_DT)
        return pvlib.temperature.sapm_module(g, ta, ws, p["a"], p["b"])
    if model == "pvsyst":
        return pvlib.temperature.pvsyst_cell(g, ta, ws, u_c=p["u_c"], u_v=p["u_v"], module_efficiency=efficiency,
                                             alpha_absorption=ALPHA)
    if model == "ross":
        return pvlib.temperature.ross(g, ta, k=p["k"])
    raise ValueError(f"unknown model {model}")


def first_order_lag(x: np.ndarray, tau_min: float, step_min: float) -> np.ndarray:
    """Zero-order-hold first-order lag of x with time constant tau (exact discretisation, stable for any step).
    NaNs in x are bridged by holding the last value."""
    x = pd.Series(np.asarray(x, dtype=float)).ffill().bfill().to_numpy()
    if tau_min <= 0:
        return x
    a = np.exp(-step_min / tau_min)
    y, _ = lfilter([1 - a], [1, -a], x, zi=[a * x[0]])
    return y


def predict(model: str, p: dict, g: pd.Series, ta: pd.Series, ws: pd.Series, efficiency: float = 0.0,
            cell: bool = False) -> pd.Series:
    """Model temperature; with p['tau'] the steady-state value is passed through a first-order lag."""
    t = pd.Series(np.asarray(steady(model, p, g, ta, ws, efficiency, cell), dtype=float), index=g.index)
    if p.get("tau"):
        step = (g.index[1] - g.index[0]).total_seconds() / 60
        t = pd.Series(first_order_lag(t.to_numpy(), p["tau"], step), index=g.index)
    return t


def prilliman(t_steady: pd.Series, ws: pd.Series, unit_mass: float = 11.7) -> pd.Series:
    """pvlib's Prilliman et al. (2020) transient model applied to a steady-state series. The model looks back
    20 minutes, so at 15-minute resolution it can use at most one previous sample. unit_mass: module mass per
    area (Sharp NU-U240F1: 19 kg / 1.63 m2)."""
    return pvlib.temperature.prilliman(t_steady, ws, unit_mass=unit_mass)


@dataclass
class FitResult:
    model: str
    params: dict
    rmse: float
    bias: float
    n: int


def fit_model(model: str, g: pd.Series, ta: pd.Series, ws: pd.Series, target: pd.Series, mask: pd.Series | None = None,
              dynamic: bool = False, efficiency: float = 0.0, cell: bool = False) -> FitResult:
    """Least-squares coefficients of `model` against `target` on the rows in `mask`. With dynamic=True a lag time
    constant tau [min] is fitted as well; the lag runs over the full series, the residual uses the mask only."""
    names = list(DEFAULTS[model])
    lo, hi = BOUNDS[model]
    x0 = [DEFAULTS[model][n] for n in names]
    if dynamic:
        names, lo, hi, x0 = names + ["tau"], lo + [0.0], hi + [120.0], x0 + [10.0]
    m = (target.notna() & g.notna() & ta.notna() & ws.notna()) if mask is None else (mask & target.notna())
    y = target[m].to_numpy()

    def resid(x):
        p = dict(zip(names, x))
        if dynamic:
            pred = predict(model, p, g, ta, ws, efficiency, cell)[m].to_numpy()
        else:
            pred = np.asarray(steady(model, p, g[m], ta[m], ws[m], efficiency, cell), dtype=float)
        return pred - y

    sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
    r = resid(sol.x)
    out = FitResult(model, dict(zip(names, map(float, sol.x))), float(np.sqrt(np.mean(r ** 2))), float(r.mean()),
                    int(m.sum()))
    log.info("%s fit%s: %s rmse %.2f K (n=%d)", model, " (dynamic)" if dynamic else "",
             {k: round(v, 4) for k, v in out.params.items()}, out.rmse, out.n)
    return out


def errors(pred: pd.Series, obs: pd.Series, mask: pd.Series) -> dict:
    e = (pred - obs)[mask].dropna()
    return {"rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()), "mae": float(e.abs().mean()),
            "n": int(len(e))}


def bias_table(pred: pd.Series, obs: pd.Series, mask: pd.Series, tz: str) -> tuple[pd.Series, pd.Series]:
    """Mean error by local hour of day and by month."""
    e = (pred - obs)[mask].dropna()
    local = e.index.tz_convert(tz)
    return e.groupby(local.hour).mean(), e.groupby(local.month).mean()
