import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_weather
from pv_thermal import thermal


def _noisy(t, sd=0.5, seed=0):
    return t + np.random.default_rng(seed).normal(0, sd, len(t))


def test_faiman_fit_recovers_known_coefficients(weather):
    w = weather
    truth = thermal.steady("faiman", {"u0": 30.0, "u1": 5.0}, w["g"], w["ta"], w["ws"])
    y = pd.Series(_noisy(np.asarray(truth)), index=w.index)
    fit = thermal.fit_model("faiman", w["g"], w["ta"], w["ws"], y, w["g"] > 50)
    assert fit.params["u0"] == pytest.approx(30.0, abs=1.0)
    assert fit.params["u1"] == pytest.approx(5.0, abs=0.3)
    assert fit.rmse < 0.6


def test_pvsyst_fit_with_explicit_efficiency(weather):
    w = weather
    truth = thermal.steady("pvsyst", {"u_c": 25.0, "u_v": 1.5}, w["g"], w["ta"], w["ws"], efficiency=0.0)
    y = pd.Series(_noisy(np.asarray(truth)), index=w.index)
    fit = thermal.fit_model("pvsyst", w["g"], w["ta"], w["ws"], y, w["g"] > 50, efficiency=0.0)
    assert fit.params["u_c"] == pytest.approx(25.0, abs=1.0)
    assert fit.params["u_v"] == pytest.approx(1.5, abs=0.3)
    # the same module at its maximum power point runs cooler by the converted share
    hot = thermal.steady("pvsyst", fit.params, 1000.0, 25.0, 1.0, efficiency=0.0)
    mpp = thermal.steady("pvsyst", fit.params, 1000.0, 25.0, 1.0, efficiency=0.15)
    assert (mpp - 25) / (hot - 25) == pytest.approx(0.85, rel=1e-6)


@pytest.mark.parametrize("model,params", [("sapm", {"a": -3.2, "b": -0.08}), ("ross", {"k": 0.025})])
def test_other_models_recover_parameters(weather, model, params):
    w = weather
    truth = thermal.steady(model, params, w["g"], w["ta"], w["ws"])
    y = pd.Series(_noisy(np.asarray(truth), 0.3), index=w.index)
    fit = thermal.fit_model(model, w["g"], w["ta"], w["ws"], y, w["g"] > 50)
    for k, v in params.items():
        assert fit.params[k] == pytest.approx(v, rel=0.05)


def test_first_order_lag_step_response():
    x = np.r_[np.zeros(5), np.ones(200)]
    y = thermal.first_order_lag(x, tau_min=10.0, step_min=1.0)
    assert y[4] == 0.0
    assert y[5 + 9] == pytest.approx(1 - np.exp(-1), abs=1e-9)  # one time constant after the step
    assert y[-1] == pytest.approx(1.0, abs=1e-6)


def test_dynamic_fit_recovers_time_constant():
    w = synthetic_weather(days=60, freq="5min")
    p = {"u0": 28.0, "u1": 4.0, "tau": 12.0}
    truth = thermal.predict("faiman", p, w["g"], w["ta"], w["ws"])
    y = pd.Series(_noisy(truth.to_numpy(), 0.3), index=w.index)
    fit = thermal.fit_model("faiman", w["g"], w["ta"], w["ws"], y, w["g"] > 50, dynamic=True)
    assert fit.params["tau"] == pytest.approx(12.0, abs=1.5)
    assert fit.params["u0"] == pytest.approx(28.0, abs=1.0)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_lagged_prediction_does_not_depend_on_datetime_resolution(unit):
    ref = synthetic_weather(unit="ns", days=3)
    w = synthetic_weather(unit=unit, days=3)
    p = {"u0": 25.0, "u1": 6.84, "tau": 8.0}
    a = thermal.predict("faiman", p, ref["g"], ref["ta"], ref["ws"]).to_numpy()
    b = thermal.predict("faiman", p, w["g"], w["ta"], w["ws"]).to_numpy()
    np.testing.assert_allclose(a, b)


def test_bias_table_groups_by_local_hour(weather):
    w = weather.iloc[:96 * 10]
    obs = pd.Series(20.0, index=w.index)
    pred = obs + 1.0
    by_hour, by_month = thermal.bias_table(pred, obs, pd.Series(True, index=w.index), "Etc/GMT+8")
    assert set(by_hour.index) == set(range(24))
    assert np.allclose(by_hour, 1.0) and np.allclose(by_month, 1.0)
