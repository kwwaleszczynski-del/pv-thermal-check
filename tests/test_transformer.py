import numpy as np
import pandas as pd
import pytest

from pv_thermal import transformer as tr


def _const(value, n=24 * 60 // 15 * 6, unit="ns"):
    idx = pd.date_range("2022-07-01", periods=n, freq="15min", tz="UTC", unit=unit)
    return pd.Series(value, index=idx, dtype=float)


def test_rated_load_at_20_degc_gives_the_98_degc_reference():
    th = tr.hotspot(_const(1.0), _const(20.0))
    assert th["hot_spot"].iloc[-1] == pytest.approx(98.0, abs=1e-6)
    assert th["top_oil"].iloc[-1] == pytest.approx(20 + 52.0, abs=1e-6)


def test_no_load_keeps_only_the_no_load_loss_rise():
    th = tr.hotspot(_const(0.0), _const(20.0))
    assert th["top_oil"].iloc[-1] == pytest.approx(20 + 52.0 * (1 / 6) ** 0.8, abs=1e-6)
    assert th["hot_spot"].iloc[-1] == pytest.approx(th["top_oil"].iloc[-1], abs=1e-9)


def test_oil_time_constant():
    k = _const(1.0, n=4 * 24)
    k.iloc[:4] = 0.0
    ta = _const(20.0, n=4 * 24)
    th = tr.hotspot(k, ta, step_min=15)
    start = 20 + 52.0 * (1 / 6) ** 0.8
    # 180 min = 12 steps after the step change (held from step 4 on), the oil has covered 63 % of the way
    rise = (th["top_oil"].iloc[4 + 11] - start) / (72.0 - start)
    assert rise == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_ageing_rates():
    assert float(tr.ageing_rate(pd.Series([98.0])).iloc[0]) == pytest.approx(1.0)
    assert float(tr.ageing_rate(pd.Series([104.0])).iloc[0]) == pytest.approx(2.0)
    assert float(tr.ageing_rate(pd.Series([110.0]), upgraded=True).iloc[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("unit", ["s", "us"])
def test_sizing_table_is_monotonic_and_resolution_free(unit):
    n = 96 * 30
    idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC", unit=unit)
    hour = (idx.hour + idx.minute / 60).to_numpy()
    p = pd.Series(np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 1000.0, index=idx)
    ta = pd.Series(25 + 5 * np.sin((hour - 9) / 24 * 2 * np.pi), index=idx)
    t = tr.sizing_table(p, ta, 1000.0, [0.6, 0.8, 1.0])
    assert t["hot_spot_max"].is_monotonic_decreasing
    assert t["ageing_days_per_year"].is_monotonic_decreasing
    assert t.loc[t["ratio"] == 1.0, "k_max"].iloc[0] == pytest.approx(1.0, abs=1e-3)
    assert tr.smallest_ratio(t) in (0.6, 0.8, 1.0)
