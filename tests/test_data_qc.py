import numpy as np
import pandas as pd
import pvlib
import pytest

from pv_thermal import data, poland, qc
from pv_thermal.site import SYSTEM_34


def _write_day(path, day, temp_f=77.0):
    t = pd.date_range(f"{day} 00:00:05", periods=96, freq="15min")
    raw = pd.DataFrame({
        "measured_on": t.strftime("%Y-%m-%d %H:%M:%S"), "ac_power_hw__2695": 500.0, "dc_power_hw__2694": 530.0,
        "dc_voltage__2682": 330.0, "dc_current__2683": 160.0, "power_factor_1000__2693": 999.0,
        "inverter_temp__2686": 54.0, "inverter_error_code__2690": 0.0, "poa_irradiance__2679": 800.0,
        "module_temp_f__2689": 122.0, "ambient_temp_f__2688": temp_f, "wind_speed_mph__2691": 10.0,
        "wind_direction__2692": 351.0, "relative_humidity__2687": 20.0, "system_id": 34})
    raw.to_csv(path, index=False)


def test_load_pvdaq_units_and_timezone(tmp_path):
    files = []
    for day in ("2016-07-01", "2016-07-02"):
        f = tmp_path / f"system_34__date_{day.replace('-', '_')}.csv"
        _write_day(f, day)
        files.append(f)
    df = data.load_pvdaq(files, "Etc/GMT+8")
    assert df.index[0] == pd.Timestamp("2016-07-01 08:00", tz="UTC")  # local midnight, UTC-8
    assert len(df) == 192 and (df.index[1] - df.index[0]) == pd.Timedelta("15min")
    row = df.iloc[0]
    assert row["p_ac"] == pytest.approx(50.0) and row["p_dc"] == pytest.approx(53.0)
    assert row["t_amb"] == pytest.approx(25.0) and row["t_mod"] == pytest.approx(50.0)
    assert row["wind"] == pytest.approx(4.4704) and row["pf"] == pytest.approx(0.999)


def test_station_on_grid_interpolates_and_keeps_long_gaps():
    hourly = pd.date_range("2016-01-01", periods=12, freq="1h", tz="UTC")
    isd = pd.DataFrame({"t_air": np.arange(12.0), "wind": 1.0}, index=hourly)
    isd.loc[hourly[5:10], "t_air"] = np.nan  # a 5-hour hole
    grid = pd.date_range("2016-01-01", periods=44, freq="15min", tz="UTC", unit="s")
    st = data.station_on_grid(isd, grid, max_gap_h=3)
    assert st["t_air"].iloc[2] == pytest.approx(0.5)
    assert st["t_air"].loc["2016-01-01 07:00":"2016-01-01 08:00"].isna().all()


def test_qc_ambient_flags_stuck_and_off_values():
    idx = pd.date_range("2016-01-01", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame({"t_amb": [10.0, qc.STUCK_C[0], -40.0, 25.0, 11.0, np.nan], "poa": 100.0}, index=idx)
    st = pd.DataFrame({"t_air": 10.0}, index=idx)
    ok, info = qc.qc_ambient(df, st)
    assert ok.tolist() == [True, False, False, False, True, False]
    assert info["stuck_share"] == pytest.approx(2 / 5)


def test_poa_scale_finds_a_miscalibrated_sensor():
    idx = pd.date_range("2016-06-01", periods=96 * 4, freq="15min", tz="UTC")
    cs = qc.clearsky_poa(idx, SYSTEM_34)["poa_cs"]
    k, per_day = qc.poa_scale(1.15 * cs, SYSTEM_34)
    assert len(per_day) >= 3
    assert k == pytest.approx(1.15, abs=0.01)


def test_poland_helpers():
    assert poland.request_weight(5, 1096) == pytest.approx(39.14, abs=0.01)
    assert poland.request_weight(2, 7) == 1.0
    idx = pd.date_range("2022-06-21", periods=24, freq="1h", tz="UTC")
    loc = pvlib.location.Location(52.23, 21.01)
    cs = loc.get_clearsky(idx - pd.Timedelta(minutes=30))
    w = pd.DataFrame({"ghi": cs["ghi"].to_numpy(), "dni": cs["dni"].to_numpy(), "dhi": cs["dhi"].to_numpy(),
                      "t_air": 20.0, "wind": 3.0}, index=idx)
    flat = poland.plane_of_array(w, 52.23, 21.01, tilt=0.0)
    day = w["ghi"].to_numpy() > 50  # Perez's horizon band differs slightly near sunset
    np.testing.assert_allclose(flat.to_numpy()[day], w["ghi"].to_numpy()[day], rtol=1e-3)
    assert (flat >= 0).all()
    pac = poland.ac_power(pd.Series([0.0, 500.0, 1200.0]), pd.Series([25.0] * 3), 1000.0, -0.004)
    assert pac.iloc[0] == 0 and pac.iloc[2] == pytest.approx(1000 / 1.2)
