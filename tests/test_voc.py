import numpy as np
import pandas as pd
import pytest

from pv_thermal import voc
from pv_thermal.site import SYSTEM_34


def test_string_voc_follows_the_datasheet_temperature_coefficient():
    t = np.array([25.0, 65.0])
    v = voc.module_voc(SYSTEM_34, np.array([1000.0, 1000.0]), t)
    assert v[0] == pytest.approx(13 * 37.4, rel=1e-3)  # datasheet Voc at STC, 13 in series
    assert (v[1] - v[0]) / 40 == pytest.approx(13 * -0.123046, rel=1e-6)
    # the pure single-diode variant falls faster with temperature (documented sensitivity case)
    vc = voc.module_voc(SYSTEM_34, np.array([1000.0, 1000.0]), t, method="cec")
    assert vc[0] == pytest.approx(v[0], rel=1e-3)
    assert (vc[1] - vc[0]) / 40 < 1.08 * 13 * -0.123046


def test_voc_thermometry_recovers_a_known_cell_temperature():
    idx = pd.date_range("2021-06-01 14:00", periods=40, freq="15min", tz="UTC", unit="s")
    g = pd.Series(np.r_[np.linspace(60, 180, 20), np.linspace(300, 1000, 20)], index=idx)
    t_back = pd.Series(np.r_[np.full(20, 25.0), np.full(20, 40.0)], index=idx)
    t_cell = t_back + np.r_[np.zeros(20), np.linspace(5, 25, 20)]  # cells hotter than the sensor in full sun
    v_dc = 0.985 * voc.module_voc(SYSTEM_34, g, t_cell)  # an aged array: 1.5 % lower voltage
    df = pd.DataFrame({"v_dc": v_dc, "t_mod": t_back}, index=idx)
    out = voc.voc_temperature(df, SYSTEM_34, g)
    assert out.attrs["scale"] == pytest.approx(0.985, abs=1e-3)
    np.testing.assert_allclose(out["t_cell_voc"].to_numpy()[20:], t_cell.to_numpy()[20:], atol=0.3)


def test_open_circuit_mask():
    df = pd.DataFrame({"p_ac": [0.0, 50.0, 0.0], "p_dc": [0.0, 52.0, 0.0], "v_dc": [450.0, 330.0, 5.0],
                       "poa": [800.0, 800.0, 800.0]})
    assert voc.open_circuit_mask(df).tolist() == [True, False, False]
