"""Inverter temperature model, cooling check and thermal derating search.

The Satcon unit's temperature channel behaves like a fan-regulated heatsink: during operation it sits at
53-56 degC whatever the air temperature, rising slightly with load. The steady model is linear,
    T_inv = c0 + c_p * P_ac / P_rated + c_a * T_air,
fitted on steps that have been operating for at least an hour (the morning warm-up is excluded). Daily residuals
go through the same CUSUM chart as the module sensor (faults.py) to flag cooling problems.

Derating: a thermally derating inverter caps its AC output below what the array could deliver. The expected AC
power is the DC performance model (irradiance, modelled cell temperature, a 30-day rolling calibration) times the
median conversion efficiency, capped at the rated output. A step is flagged when AC power is more than 8 % below
that, flat (a plateau) and below rated power.

Usage:
    coef = fit_inverter(df, mask, p_rated)
    exp_ac = expected_ac(df, t_cell, plant)
    flags = derating_flags(df["p_ac"], exp_ac, operating)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pv_thermal.site import Plant

log = logging.getLogger(__name__)


def warm_mask(df: pd.DataFrame, min_steps: int = 4) -> pd.Series:
    """Operating steps preceded by at least min_steps operating steps (excludes the morning warm-up)."""
    run = df["operating"].astype(int).rolling(min_steps + 1, min_periods=min_steps + 1).sum()
    return (run == min_steps + 1) & df["t_inv"].notna() & df["ta"].notna()


def fit_inverter(df: pd.DataFrame, mask: pd.Series, p_rated: float) -> dict:
    x = np.column_stack([np.ones(int(mask.sum())), df.loc[mask, "p_ac"] / p_rated, df.loc[mask, "ta"]])
    beta, *_ = np.linalg.lstsq(x, df.loc[mask, "t_inv"].to_numpy(float), rcond=None)
    coef = {"c0": float(beta[0]), "c_p": float(beta[1]), "c_a": float(beta[2])}
    pred = predict_inverter(df, coef, p_rated)[mask]
    coef["rmse"] = float(np.sqrt(((pred - df.loc[mask, "t_inv"]) ** 2).mean()))
    log.info("inverter: T = %.2f + %.2f P/Pr + %.3f Ta, rmse %.2f K", coef["c0"], coef["c_p"], coef["c_a"],
             coef["rmse"])
    return coef


def predict_inverter(df: pd.DataFrame, coef: dict, p_rated: float) -> pd.Series:
    return coef["c0"] + coef["c_p"] * df["p_ac"] / p_rated + coef["c_a"] * df["ta"]


def daily_residual(df: pd.DataFrame, pred: pd.Series, mask: pd.Series, tz: str, min_points: int = 8
                   ) -> pd.DataFrame:
    r = (df["t_inv"] - pred)[mask].dropna()
    day = r.index.tz_convert(tz).normalize().tz_localize(None)
    d = r.groupby(day).agg(["mean", "size"])
    return d[d["size"] >= min_points].rename(columns={"mean": "resid", "size": "n"})


def expected_ac(df: pd.DataFrame, t_cell: pd.Series, plant: Plant, window: str = "30D") -> tuple[pd.Series, float]:
    """Expected AC power [kW] and the median DC-to-AC efficiency used for it. The rolling calibration (soiling,
    degradation, sensor scale) uses only part-load steps (20-60 % of rated DC power), which a derating cap near
    full power cannot pull down."""
    p_model = plant.pdc0_kw * df["g"] / 1000 * (1 + plant.gamma_pdc * (t_cell - 25))
    part = df["operating"] & (p_model > 0.2 * plant.pdc0_kw) & (p_model < 0.6 * plant.pdc0_kw)
    k = (df["p_dc"] / p_model).where(part).rolling(window, center=True, min_periods=100).median()
    eff = float((df["p_ac"] / df["p_dc"]).where(df["operating"] & (df["p_dc"] > 20)).median())
    return np.minimum(eff * k * p_model, 0.992 * plant.pac0_kw), eff


def derating_flags(p_ac: pd.Series, exp_ac: pd.Series, mask: pd.Series, p_rated: float, shortfall: float = 0.08,
                   flat_kw: float = 0.5) -> pd.Series:
    """Plateau below expected and below rated power: |dP| < flat_kw to both neighbours and P < (1 - shortfall) E."""
    flat = (p_ac.diff().abs() < flat_kw) & (p_ac.diff(-1).abs() < flat_kw)
    return mask & flat & (p_ac < (1 - shortfall) * exp_ac) & (p_ac < 0.96 * p_rated)


def inject_derating(df: pd.DataFrame, cap_kw: float, t_air_above: float) -> pd.DataFrame:
    """Cap AC power at cap_kw whenever the air is warmer than t_air_above (an outdoor inverter's derating curve,
    simplified to one step). DC power is reduced in proportion, as the inverter leaves the MPP."""
    out = df.copy()
    hit = (out["ta"] > t_air_above) & (out["p_ac"] > cap_kw)
    scale = cap_kw / out.loc[hit, "p_ac"]
    out.loc[hit, "p_ac"] = cap_kw
    out.loc[hit, "p_dc"] *= scale
    out.attrs["derated_steps"] = hit
    return out
