import numpy as np
import pandas as pd
import pytest

from pv_thermal import performance, thermal

GAMMA = -0.0045


def test_temperature_loss_of_a_constant_cell_temperature(weather):
    tc = pd.Series(35.0, index=weather.index)
    assert performance.temperature_loss(weather["g"], tc, GAMMA, None) == pytest.approx(GAMMA * 10)
    monthly = performance.temperature_loss(weather["g"], tc, GAMMA)
    assert len(monthly) == 12 and np.allclose(monthly, GAMMA * 10)


def test_pr_correction_removes_artificial_seasonality(weather):
    w = weather
    tc = thermal.predict("faiman", {"u0": 25.0, "u1": 6.84}, w["g"], w["ta"], w["ws"])
    pdc0, pr_true = 100.0, 0.85
    power = pdc0 * w["g"] / 1000 * pr_true * (1 + GAMMA * (tc - 25))
    mask = w["g"] > 50
    pr = performance.monthly_pr(power, w["g"], tc, pdc0, GAMMA, mask)
    raw = performance.seasonality(pr["pr_raw"], pr["t_cell_w"])
    stc = performance.seasonality(pr["pr_stc"], pr["t_cell_w"])
    nrel = performance.seasonality(pr["pr_nrel"], pr["t_cell_w"])
    assert raw["spread"] > 0.03 and raw["corr_t"] < -0.9  # hot months look worse
    assert stc["spread"] < 1e-9 and np.allclose(pr["pr_stc"], pr_true)
    # NREL divides by 1 + gamma (Tc - T_typ) instead of 1 + gamma (Tc - 25): a first-order correction that leaves a
    # small residual swing
    assert nrel["spread"] < 0.2 * raw["spread"]
    # NREL's level sits at the plant's typical cell temperature instead of 25 degC
    assert pr["pr_nrel"].mean() == pytest.approx(pr_true * (1 + GAMMA * (pr.attrs["t_typ"] - 25)), rel=5e-3)


def test_cell_from_back_adds_three_kelvin_at_1000():
    assert performance.cell_from_back(pd.Series([40.0]), pd.Series([1000.0])).iloc[0] == pytest.approx(43.0)
