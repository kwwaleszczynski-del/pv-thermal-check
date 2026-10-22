"""Electrical cell temperature from open-circuit voltage (the idea behind the IEC 60904-5 equivalent cell
temperature): while the inverter is down the array sits at open circuit, the DC voltage channel then reads the
string Voc, and Voc depends on cell temperature through a well-known coefficient (-0.123 V/K for the Sharp
NU-U240F1) and only logarithmically on irradiance. With no current flowing there is no wiring drop either.

Calibration: one array-level scale factor, taken where cell and back-sheet must agree (low irradiance, little
heating). Inversion: for each step, the cell temperature at which the Voc model (module_voc) reproduces the
measured Voc at the measured irradiance.

Usage:
    out = voc_temperature(df_outage, plant, poa_true)   # columns t_cell_voc, scale
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import pvlib

from pv_thermal.site import Plant

log = logging.getLogger(__name__)

T_GRID = np.arange(-20.0, 110.0, 0.5)


@lru_cache
def cec_parameters(module: str) -> tuple[float, ...]:
    m = pvlib.pvsystem.retrieve_sam("CECMod")[module]
    return tuple(float(m[k]) for k in ("alpha_sc", "a_ref", "I_L_ref", "I_o_ref", "R_sh_ref", "R_s", "Adjust"))


@lru_cache
def datasheet_beta_voc(module: str) -> float:
    """Voc temperature coefficient [V/K per module] as listed in the CEC database (the datasheet value)."""
    return float(pvlib.pvsystem.retrieve_sam("CECMod")[module]["beta_oc"])


def module_voc(plant: Plant, g, t_cell, method: str = "datasheet") -> np.ndarray:
    """String open-circuit voltage [V].
    method='datasheet': irradiance dependence from the CEC single-diode model at 25 degC, temperature dependence
    from the datasheet coefficient (the IEC 60904-5 practice). method='cec': the single-diode model throughout;
    for this module its Voc falls about 13 % faster with temperature than the datasheet says."""
    g, t_cell = np.asarray(g, float), np.asarray(t_cell, float)
    params = cec_parameters(plant.module)
    if method == "cec":
        p = pvlib.pvsystem.calcparams_cec(g, t_cell, *params)
        return plant.modules_series * np.asarray(pvlib.pvsystem.singlediode(*p)["v_oc"], dtype=float)
    p = pvlib.pvsystem.calcparams_cec(g, np.full_like(g, 25.0), *params)
    v25 = np.asarray(pvlib.pvsystem.singlediode(*p)["v_oc"], dtype=float)
    return plant.modules_series * (v25 + datasheet_beta_voc(plant.module) * (t_cell - 25.0))


def open_circuit_mask(df: pd.DataFrame, min_g: float = 20.0) -> pd.Series:
    """Steps where the inverter exports nothing but the DC voltage channel shows a lit array."""
    return (df["p_ac"].abs() < 1) & (df["p_dc"].abs() < 1) & (df["v_dc"] > 200) & (df["poa"] > min_g)


def voc_temperature(df: pd.DataFrame, plant: Plant, poa: pd.Series, calib_g: tuple[float, float] = (50, 200),
                    method: str = "datasheet") -> pd.DataFrame:
    """Cell temperature from the open-circuit DC voltage on the rows of `df` (already restricted to open circuit).
    `poa` is the irradiance on the true scale (sensor reading divided by the clear-sky scale factor)."""
    g = poa.reindex(df.index)
    low = (g >= calib_g[0]) & (g < calib_g[1]) & df["t_mod"].notna()
    scale = float(np.median(df.loc[low, "v_dc"] / module_voc(plant, g[low], df.loc[low, "t_mod"], method)))
    gv, vv = g.to_numpy(float), df["v_dc"].to_numpy(float)
    ok = np.isfinite(gv) & np.isfinite(vv) & (gv > 0)
    # Voc on an (irradiance x temperature) grid in one call, then invert row by row (Voc falls with T)
    curves = scale * module_voc(plant, np.repeat(gv[ok], len(T_GRID)), np.tile(T_GRID, ok.sum()), method)
    curves = curves.reshape(ok.sum(), len(T_GRID))
    t_cell = np.full(len(df), np.nan)
    t_cell[ok] = [np.interp(-v, -c, T_GRID, left=np.nan, right=np.nan) for v, c in zip(vv[ok], curves)]
    log.info("Voc thermometry on %d open-circuit steps; calibration scale %.4f from %d low-irradiance steps",
             len(df), scale, int(low.sum()))
    out = pd.DataFrame({"t_cell_voc": t_cell, "g": g, "t_mod": df["t_mod"], "v_dc": df["v_dc"]}, index=df.index)
    out.attrs["scale"] = scale
    out.attrs["n_calib"] = int(low.sum())
    return out
