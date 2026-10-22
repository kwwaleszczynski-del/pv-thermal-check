"""Plant and site constants.

PVDAQ system 34 (Andre Agassi Preparatory Academy, Building A, Las Vegas NV, from the PVDAQ system metadata):
611 Sharp NU-U240F1 mono-Si modules (13 in series x 47 strings, 146.64 kWp) on a flat roof, tilt 11.2 deg south,
one Satcon 135 kW central inverter, in service since 2010. Timestamps are local standard time (UTC-8, no DST).
The independent weather reference is the NOAA ISD station at Las Vegas airport (KLAS, USAF 723860 / WBAN 23169),
about 14 km south-east of the plant, with 2 m air temperature and 10 m wind.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plant:
    latitude: float
    longitude: float
    altitude: float
    tz: str
    tilt: float
    azimuth: float
    pdc0_kw: float
    pac0_kw: float
    modules_series: int
    strings: int
    module: str  # key in the pvlib CEC module database
    gamma_pdc: float  # power temperature coefficient [1/K]
    module_efficiency: float  # STC efficiency [-]


SYSTEM_34 = Plant(latitude=36.1952, longitude=-115.1582, altitude=620.0, tz="Etc/GMT+8", tilt=11.2, azimuth=180.0,
                  pdc0_kw=146.64, pac0_kw=135.0, modules_series=13, strings=47, module="Sharp_NU_U240F1",
                  gamma_pdc=-0.00458, module_efficiency=0.147)

PVDAQ_SYSTEM = 34
ISD_STATION = (723860, 23169)  # Las Vegas McCarran (KLAS)
YEARS = (2016, 2017)

# Polish sites for the transfer: coast, west, centre, north-east, south-west, south-east
POLAND = {
    "Gdansk": (54.35, 18.65),
    "Szczecin": (53.43, 14.55),
    "Warsaw": (52.23, 21.01),
    "Bialystok": (53.13, 23.16),
    "Wroclaw": (51.11, 17.03),
    "Rzeszow": (50.04, 22.00),
}
POLAND_YEARS = (2022, 2024)  # inclusive ERA5 range
POLAND_TILT = 30.0  # typical fixed-tilt ground mount in Poland
