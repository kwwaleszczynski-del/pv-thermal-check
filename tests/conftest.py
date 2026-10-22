"""Synthetic inputs shared by the tests: a year of 15-minute weather built offline from pvlib's clear-sky model,
with a deterministic cloud factor, seasonal air temperature and random wind. No network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import pytest

from pv_thermal.site import SYSTEM_34


def synthetic_weather(unit: str = "ns", seed: int = 1, start: str = "2021-01-01", days: int = 365,
                      freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=days * int(pd.Timedelta("1D") / pd.Timedelta(freq)), freq=freq, tz="UTC",
                        unit=unit)
    p = SYSTEM_34
    loc = pvlib.location.Location(p.latitude, p.longitude, altitude=p.altitude)
    sp = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx, solar_position=sp)
    poa = pvlib.irradiance.get_total_irradiance(p.tilt, p.azimuth, sp["apparent_zenith"], sp["azimuth"], cs["dni"],
                                                cs["ghi"], cs["dhi"])["poa_global"].fillna(0).clip(lower=0)
    rng = np.random.default_rng(seed)
    day = np.floor((idx - idx[0]).total_seconds() / 86400).astype(int)
    cloud = np.clip(rng.uniform(0.3, 1.05, day.max() + 1), 0, 1)[day]
    doy = idx.dayofyear.to_numpy()
    hour = (idx.hour + idx.minute / 60).to_numpy()
    ta = 20 - 12 * np.cos(2 * np.pi * (doy - 15) / 365) + 6 * np.sin(2 * np.pi * (hour - 16) / 24)
    ws = rng.gamma(2.0, 1.5, len(idx))
    return pd.DataFrame({"g": poa.to_numpy() * cloud, "ta": ta, "ws": ws}, index=idx)


@pytest.fixture(scope="session")
def weather() -> pd.DataFrame:
    return synthetic_weather()
