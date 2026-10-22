import numpy as np
import pandas as pd
import pytest

from pv_thermal import inverter


def _frame(n=96 * 20):
    idx = pd.date_range("2022-07-01", periods=n, freq="15min", tz="UTC")
    hour = (idx.hour + idx.minute / 60).to_numpy()
    p = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 120.0
    ta = 25 + 10 * np.sin((hour - 9) / 24 * 2 * np.pi)
    df = pd.DataFrame({"p_ac": p, "p_dc": p / 0.95, "ta": ta, "g": p / 120 * 1000}, index=idx)
    df["operating"] = df["p_ac"] > 1
    return df


def test_fit_recovers_coefficients():
    df = _frame()
    rng = np.random.default_rng(0)
    df["t_inv"] = 50 + 4 * df["p_ac"] / 135 + 0.1 * df["ta"] + rng.normal(0, 0.2, len(df))
    mask = inverter.warm_mask(df)
    coef = inverter.fit_inverter(df, mask, 135.0)
    assert coef["c0"] == pytest.approx(50, abs=0.5)
    assert coef["c_p"] == pytest.approx(4, abs=0.3)
    assert coef["c_a"] == pytest.approx(0.1, abs=0.02)
    assert coef["rmse"] < 0.3


def test_warm_mask_skips_the_morning_warm_up():
    df = _frame(96)
    df["t_inv"] = 50.0
    mask = inverter.warm_mask(df, min_steps=4)
    first_on = int(np.argmax(df["operating"].to_numpy()))
    assert not mask.iloc[first_on:first_on + 4].any() and mask.iloc[first_on + 4]


def test_derating_cap_is_flagged():
    df = _frame()
    capped = inverter.inject_derating(df, cap_kw=90.0, t_air_above=30.0)
    hit = capped.attrs["derated_steps"]
    assert hit.any() and (capped.loc[hit, "p_ac"] == 90.0).all()
    assert np.allclose(capped.loc[hit, "p_ac"] / capped.loc[hit, "p_dc"], 0.95)
    flags = inverter.derating_flags(capped["p_ac"], df["p_ac"], df["operating"], 135.0)
    real_loss = hit & (df["p_ac"] > 90 / 0.92)
    inner = real_loss & real_loss.shift(1, fill_value=False) & real_loss.shift(-1, fill_value=False)
    assert flags[inner].all()
    assert not inverter.derating_flags(df["p_ac"], df["p_ac"], df["operating"], 135.0).any()
