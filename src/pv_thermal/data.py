"""Open data for the plant: NREL PVDAQ (the plant itself) and NOAA ISD-Lite (an independent weather station).
Everything is downloaded once into a cache directory. The Polish weather (ERA5) is fetched in poland.py.

Sources and licences:
- NREL PVDAQ: Deline et al. (2021), Photovoltaic Data Acquisition (PVDAQ) Public Datasets, OEDI,
  https://doi.org/10.25984/1846021, CC BY 4.0. One CSV per system and day.
- NOAA ISD-Lite: hourly subset of the Integrated Surface Database, NOAA NCEI, US public domain.

Usage:
    from pv_thermal.data import fetch_pvdaq, load_pvdaq, load_isd
    files = fetch_pvdaq(34, "2016-01-01", "2017-12-31", Path("data/pvdaq"))
    plant = load_pvdaq(files, "Etc/GMT+8")            # 15-min UTC frame, SI units, raw sensor values
    station = load_isd(723860, 23169, [2016, 2017], Path("data/isd"))
"""
from __future__ import annotations

import gzip
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PVDAQ_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"
ISD_URL = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data"

PVDAQ_COLUMNS = {
    "ac_power_hw__2695": "p_ac", "dc_power_hw__2694": "p_dc", "dc_voltage__2682": "v_dc", "dc_current__2683": "i_dc",
    "power_factor_1000__2693": "pf", "inverter_temp__2686": "t_inv", "inverter_error_code__2690": "status",
    "poa_irradiance__2679": "poa", "module_temp_f__2689": "t_mod", "ambient_temp_f__2688": "t_amb",
    "wind_speed_mph__2691": "wind", "wind_direction__2692": "wind_dir", "relative_humidity__2687": "rh",
}


def _download(url: str, path: Path, timeout: int = 60) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        path.write_bytes(r.read())


def fetch_pvdaq(system_id: int, start: str, end: str, data_dir: Path, workers: int = 8) -> list[Path]:
    """Daily PVDAQ files from `start` to `end` (inclusive) in data_dir/system_<id>; existing files are reused."""
    out = Path(data_dir) / f"system_{system_id}"
    out.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    paths = [out / f"system_{system_id}__date_{d:%Y_%m_%d}.csv" for d in days]

    def get(job):
        day, path = job
        if path.exists():
            return False
        _download(f"{PVDAQ_URL}/system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
                  f"{path.name}", path)
        return True

    with ThreadPoolExecutor(workers) as pool:
        new = sum(pool.map(get, zip(days, paths)))
    log.info("PVDAQ system %d: %d daily files (%d downloaded) in %s", system_id, len(paths), new, out)
    return paths


def _f_to_c(x: pd.Series) -> pd.Series:
    return (x - 32) * 5 / 9


def load_pvdaq(files: list[Path], tz: str) -> pd.DataFrame:
    """Daily CSV files -> regular 15-min UTC grid in SI units. Sensor values are converted but NOT cleaned:
    the exact-0 degF and -40 degF readings stay in (as -17.8 and -40 degC) so that qc.py can flag them.

    Columns: p_ac, p_dc [kW]; v_dc [V]; i_dc [A]; pf [-]; t_inv [degC]; status (inverter error code);
    poa [W/m2]; t_mod, t_amb [degC]; wind [m/s]; wind_dir [deg]; rh [%]."""
    raw = pd.concat([pd.read_csv(f).reindex(columns=["measured_on", *PVDAQ_COLUMNS]) for f in files],
                    ignore_index=True)
    t = pd.to_datetime(raw["measured_on"]).dt.round("15min").dt.tz_localize(tz).dt.tz_convert("UTC")
    raw = raw.rename(columns=PVDAQ_COLUMNS).set_index(t).drop(columns="measured_on")
    raw = raw[~raw.index.duplicated()]
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    raw = raw.reindex(grid)
    df = pd.DataFrame({
        "p_ac": raw["p_ac"] * 0.1,  # hectowatt -> kW
        "p_dc": raw["p_dc"] * 0.1,
        "v_dc": raw["v_dc"],
        "i_dc": raw["i_dc"],
        "pf": raw["pf"] / 1000,
        "t_inv": raw["t_inv"],  # integer degC
        "status": raw["status"],
        "poa": raw["poa"],
        "t_mod": _f_to_c(raw["t_mod"]),
        "t_amb": _f_to_c(raw["t_amb"]),
        "wind": raw["wind"] * 0.44704,  # mph -> m/s
        "wind_dir": raw["wind_dir"],
        "rh": raw["rh"],
    }, index=grid)
    df.index.name = "time"
    log.info("PVDAQ: %d steps %s .. %s", len(df), grid[0], grid[-1])
    return df


def load_isd(usaf: int, wban: int, years: list[int], data_dir: Path) -> pd.DataFrame:
    """NOAA ISD-Lite hourly observations (UTC): air temperature t_air [degC], dew point [degC] and wind [m/s]."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in years:
        path = out / f"{usaf}-{wban}-{y}.gz"
        if not path.exists():
            _download(f"{ISD_URL}/{y}/{path.name}", path)
            log.info("downloaded %s", path.name)
        with gzip.open(path, "rt") as f:
            frames.append(pd.read_csv(f, sep=r"\s+", header=None, usecols=range(9),
                                      names=["y", "m", "d", "h", "t_air", "dew", "slp", "wdir", "wind"]))
    isd = pd.concat(frames, ignore_index=True).replace(-9999, np.nan)
    idx = pd.to_datetime(dict(year=isd.y, month=isd.m, day=isd.d, hour=isd.h)).dt.tz_localize("UTC")
    df = pd.DataFrame({"t_air": isd["t_air"].to_numpy() / 10, "dew": isd["dew"].to_numpy() / 10,
                       "wind": isd["wind"].to_numpy() / 10}, index=pd.DatetimeIndex(idx))
    df = df[~df.index.duplicated()].sort_index()
    log.info("ISD %d-%d: %d hourly records", usaf, wban, len(df))
    return df


def station_on_grid(isd: pd.DataFrame, grid: pd.DatetimeIndex, max_gap_h: float = 3) -> pd.DataFrame:
    """Hourly station data linearly interpolated onto a finer grid. A grid point is left NaN when the valid
    observations on either side of it are more than max_gap_h apart (so long gaps stay empty end to end)."""
    full = isd.reindex(isd.index.union(grid))
    out = full.interpolate("time", limit_area="inside").reindex(grid)
    t = pd.Series(full.index, index=full.index)
    for col in out.columns:
        valid = t.where(full[col].notna())
        span = (valid.bfill() - valid.ffill()).reindex(grid)
        out.loc[(span > pd.Timedelta(hours=max_gap_h)).to_numpy(), col] = np.nan
    return out

