"""Transfer to Polish weather: the fitted thermal coefficients applied to hourly ERA5 reanalysis (via the
Open-Meteo archive API, CC BY 4.0) at several Polish sites.

The PV model is deliberately simple and identical for every site: fixed-tilt south-facing array, POA from
GHI/DNI/DHI with the Perez transposition, cell temperature from a thermal model, a linear power temperature
coefficient. It answers one question: how big are temperature losses, and how much do the thermal models disagree,
in Polish weather rather than in Las Vegas.

Open-Meteo hourly radiation is the mean over the preceding hour, so the sun position is taken at the middle of
that hour. ERA5 wind is at 10 m, the same height as the airport anemometer the coefficients were fitted with.

Usage:
    w = fetch_era5(sites, "2022-01-01", "2024-12-31", Path("data/era5"))   # {site: hourly DataFrame}
    poa = plane_of_array(w["Warsaw"], lat, lon, tilt=30)
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "diffuse_radiation",
             "direct_normal_irradiance"]


def request_weight(n_vars: int, n_days: int) -> float:
    """Open-Meteo's cost of one location: max(1, vars/10 * max(1, days/14))."""
    return max(1.0, n_vars / 10 * max(1.0, n_days / 14))


def fetch_era5(sites: dict[str, tuple[float, float]], start: str, end: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Hourly ERA5 for each site, cached as one JSON per site. Missing sites are fetched in a single request."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = {s: out / f"era5_{s}_{start}_{end}.json" for s in sites}
    todo = [s for s in sites if not path[s].exists()]
    if todo:
        n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        log.info("Open-Meteo ERA5 request for %d sites, weight %.1f", len(todo),
                 len(todo) * request_weight(len(VARIABLES), n_days))
        q = {"latitude": ",".join(f"{sites[s][0]}" for s in todo),
             "longitude": ",".join(f"{sites[s][1]}" for s in todo),
             "start_date": start, "end_date": end, "hourly": ",".join(VARIABLES), "models": "era5",
             "wind_speed_unit": "ms", "timezone": "GMT"}
        with urllib.request.urlopen(f"{ARCHIVE_URL}?{urllib.parse.urlencode(q)}", timeout=300) as r:
            js = json.loads(r.read())
        for s, part in zip(todo, js if isinstance(js, list) else [js]):
            path[s].write_text(json.dumps(part))
    frames = {}
    for s in sites:
        h = json.loads(path[s].read_text())["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(h["time"]), tz="UTC")
        frames[s] = pd.DataFrame({"t_air": h["temperature_2m"], "wind": h["wind_speed_10m"],
                                  "ghi": h["shortwave_radiation"], "dhi": h["diffuse_radiation"],
                                  "dni": h["direct_normal_irradiance"]}, index=idx).astype(float)
    return frames


def plane_of_array(w: pd.DataFrame, lat: float, lon: float, tilt: float, azimuth: float = 180.0) -> pd.Series:
    """POA [W/m2] from hourly-mean GHI/DNI/DHI (sun position at mid-hour, Perez transposition)."""
    mid = w.index - pd.Timedelta(minutes=30)
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
    dni_extra = pvlib.irradiance.get_extra_radiation(mid)
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"].to_numpy())
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy(), w["dni"].to_numpy(),
        w["ghi"].to_numpy(), w["dhi"].to_numpy(), dni_extra=dni_extra.to_numpy(), airmass=am, model="perez",
        albedo=0.2)["poa_global"]
    return pd.Series(np.nan_to_num(np.asarray(poa, float)).clip(min=0), index=w.index)


def ac_power(poa: pd.Series, t_cell: pd.Series, pdc0: float, gamma: float, dc_ac: float = 1.2,
             losses: float = 0.10, eff: float = 0.98) -> pd.Series:
    """AC power [kW] of an array of pdc0 kWp with inverter rating pdc0 / dc_ac (clipped)."""
    p_dc = pdc0 * poa / 1000 * (1 + gamma * (t_cell - 25)) * (1 - losses)
    return np.minimum(eff * p_dc, pdc0 / dc_ac).clip(lower=0)
