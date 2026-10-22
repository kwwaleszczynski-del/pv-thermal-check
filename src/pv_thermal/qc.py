"""Sensor quality control against independent references.

Three checks, each producing a flag column and a small summary:
- ambient temperature: stuck values (exact 0 degF / -40 degF) and disagreement with the airport station;
- wind: level and correlation against the station's 10 m anemometer;
- plane-of-array irradiance: scale factor against a clear-sky model on days the clear-sky detector accepts.

Usage:
    flags, summary = qc_ambient(plant_df, station_on_grid_df)
    k, per_day = poa_scale(plant_df["poa"], plant)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pvlib

from pv_thermal.site import Plant

log = logging.getLogger(__name__)

STUCK_C = ((0 - 32) * 5 / 9, -40.0)  # 0 degF and -40 degF after conversion


def qc_ambient(df: pd.DataFrame, station: pd.DataFrame, max_dev: float = 8.0) -> tuple[pd.Series, dict]:
    """Boolean series 'site ambient usable' and a summary. A reading is rejected when it is one of the stuck
    values or deviates from the interpolated station temperature by more than max_dev kelvin."""
    t = df["t_amb"]
    stuck = pd.Series(np.isclose(t, STUCK_C[0], atol=1e-6) | np.isclose(t, STUCK_C[1], atol=1e-6), index=t.index)
    dev = (t - station["t_air"]).abs()
    ok = t.notna() & ~stuck & (dev <= max_dev)
    diff = (t - station["t_air"])[ok]
    day = df["poa"] > 50
    summary = {
        "stuck_share": float(stuck[t.notna()].mean()),
        "stuck_share_by_month": stuck[t.notna()].groupby(t.index[t.notna()].month).mean().round(3).to_dict(),
        "outlier_share": float(((dev > max_dev) & ~stuck)[t.notna()].mean()),
        "usable_share": float(ok[t.notna()].mean()),
        "bias_day_k": float(diff[day].mean()),
        "bias_night_k": float(diff[~day].mean()),
        "sd_k": float(diff.std()),
    }
    log.info("site ambient: %.0f %% stuck, %.0f %% usable; bias vs station day %+.1f K, night %+.1f K",
             100 * summary["stuck_share"], 100 * summary["usable_share"], summary["bias_day_k"],
             summary["bias_night_k"])
    return ok, summary


def qc_wind(df: pd.DataFrame, station: pd.DataFrame) -> dict:
    """Compare the site anemometer and vane with the station (hourly means)."""
    h = pd.DataFrame({"site": df["wind"], "station": station["wind"]}).resample("1h").mean().dropna()
    return {
        "site_mean_ms": float(h["site"].mean()),
        "station_mean_ms": float(h["station"].mean()),
        "corr_hourly": float(h["site"].corr(h["station"])),
        "vane_iqr_deg": float(df["wind_dir"].quantile(0.75) - df["wind_dir"].quantile(0.25)),
    }


def clearsky_poa(index: pd.DatetimeIndex, plant: Plant) -> pd.DataFrame:
    loc = pvlib.location.Location(plant.latitude, plant.longitude, altitude=plant.altitude)
    sp = loc.get_solarposition(index)
    cs = loc.get_clearsky(index, solar_position=sp)
    poa = pvlib.irradiance.get_total_irradiance(plant.tilt, plant.azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                cs["dni"], cs["ghi"], cs["dhi"], albedo=0.2)["poa_global"]
    return pd.DataFrame({"poa_cs": poa, "elevation": sp["apparent_elevation"]})


def poa_scale(poa: pd.Series, plant: Plant, min_cs: float = 300.0) -> tuple[float, pd.Series]:
    """Median ratio measured / clear-sky POA over the clear part of days that pvlib's detector marks clear for at
    least 90 % of the time with clear-sky POA above min_cs. Returns (factor, per-day ratios)."""
    cs = clearsky_poa(poa.index, plant)["poa_cs"]
    ratios = {}
    for day, g in poa.groupby(poa.index.date):
        g = g.dropna()
        if len(g) < 90:
            continue
        c = cs.reindex(g.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clear = pvlib.clearsky.detect_clearsky(g, c, infer_limits=True)
        bright = c > min_cs
        if bright.sum() > 10 and (clear & bright).sum() >= 0.9 * bright.sum():
            ratios[pd.Timestamp(day)] = float((g[clear & bright] / c[clear & bright]).median())
    per_day = pd.Series(ratios, name="poa_ratio")
    k = float(per_day.median())
    log.info("POA sensor / clear-sky model on %d clear days: %.3f", len(per_day), k)
    return k, per_day
