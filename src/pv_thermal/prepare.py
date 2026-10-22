"""Assemble the analysis frame: PVDAQ system 34 on a 15-min UTC grid, the airport station interpolated onto it,
sensor QC flags and the irradiance on the clear-sky scale.

Column conventions in the returned frame:
    g_sensor   POA sensor reading [W/m2]
    g          POA divided by the sensor's clear-sky scale factor (best estimate of true POA)
    ta, ws     station air temperature [degC] and 10 m wind [m/s] (the model inputs)
    ta_site    site ambient sensor where it passes QC, else NaN
    t_mod      back-of-module sensor [degC]
    operating  inverter exporting power;  open_circuit  inverter down, array lit and DC voltage present

Usage:
    df, info = build_frame(Path("data"), pvdaq_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pv_thermal import data, qc, voc
from pv_thermal.site import ISD_STATION, PVDAQ_SYSTEM, SYSTEM_34, YEARS

log = logging.getLogger(__name__)


def build_frame(data_dir: Path, pvdaq_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    plant = SYSTEM_34
    pvdaq_dir = Path(pvdaq_dir) if pvdaq_dir else Path(data_dir) / "pvdaq"
    files = data.fetch_pvdaq(PVDAQ_SYSTEM, f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", pvdaq_dir)
    raw = data.load_pvdaq(files, plant.tz)
    isd = data.load_isd(*ISD_STATION, list(YEARS), Path(data_dir) / "isd")
    st = data.station_on_grid(isd, raw.index)

    amb_ok, amb_info = qc.qc_ambient(raw, st)
    wind_info = qc.qc_wind(raw, st)
    k_poa, poa_days = qc.poa_scale(raw["poa"].clip(lower=0), plant)

    df = pd.DataFrame({
        "g_sensor": raw["poa"].clip(lower=0),
        "g": raw["poa"].clip(lower=0) / k_poa,
        "ta": st["t_air"],
        "ws": st["wind"],
        "ta_site": raw["t_amb"].where(amb_ok),
        "ws_site": raw["wind"],
        "t_mod": raw["t_mod"].where(~np.isclose(raw["t_mod"], qc.STUCK_C[0], atol=1e-6)),
        "t_inv": raw["t_inv"],
        "p_ac": raw["p_ac"],
        "p_dc": raw["p_dc"],
        "v_dc": raw["v_dc"],
        "i_dc": raw["i_dc"],
        "pf": raw["pf"],
        "status": raw["status"],
    }, index=raw.index)
    df["operating"] = (df["p_ac"] > 1.0) & (df["p_dc"] > 1.0)
    df["open_circuit"] = voc.open_circuit_mask(raw.assign(poa=df["g"]))
    info = {"ambient": amb_info, "wind": wind_info, "poa_scale": k_poa, "poa_clear_days": int(len(poa_days)),
            "poa_ratio_by_month": poa_days.groupby(poa_days.index.month).median().round(3).to_dict()}
    log.info("frame: %d steps, %d operating daylight steps, %d open-circuit daylight steps", len(df),
             int((df["operating"] & (df["g"] > 50)).sum()), int(df["open_circuit"].sum()))
    return df, info
