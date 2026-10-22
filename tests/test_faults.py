import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_weather
from pv_thermal import faults, thermal

TZ = "Etc/GMT+8"


@pytest.fixture(scope="module")
def two_years():
    w = synthetic_weather(days=730, seed=3)
    model = thermal.predict("faiman", {"u0": 30.0, "u1": 5.0}, w["g"], w["ta"], w["ws"])
    rng = np.random.default_rng(5)
    daily_err = rng.normal(0, 0.8, 731)[np.floor((w.index - w.index[0]).total_seconds() / 86400).astype(int)]
    sensor = model + daily_err + rng.normal(0, 1.0, len(w))
    mask = w["g"] > 200
    return w, model, sensor, mask


def test_daily_features_of_a_half_contact_sensor(weather):
    w = weather.iloc[: 96 * 30]
    model = thermal.predict("faiman", {"u0": 30.0, "u1": 5.0}, w["g"], w["ta"], w["ws"])
    sensor = w["ta"] + 0.5 * (model - w["ta"])
    f = faults.daily_features(sensor, model, w["ta"], w["g"] > 200, TZ)
    assert len(f) > 20
    assert np.allclose(f["ratio"], 0.5) and np.allclose(f["gain"].dropna(), 0.5)
    assert np.allclose(f["offset"].dropna(), 0.0, atol=1e-9)


def test_inject_kinds(weather):
    w = weather.iloc[: 96 * 200]
    t = w["ta"] + w["g"] / 30
    start = w.index[96 * 100]
    after = w.index >= start
    lit = after & (w["g"] > 20).to_numpy()
    assert np.allclose((faults.inject(t, w["ta"], w["g"], "offset_plus", start) - t)[after], 3.0)
    assert np.allclose((faults.inject(t, w["ta"], w["g"], "offset_minus", start) - t)[~after], 0.0)
    drift = faults.inject(t, w["ta"], w["g"], "drift", start) - t
    assert drift.iloc[-1] == pytest.approx(4.5 * (w.index[-1] - start).total_seconds() / 86400 / 90)
    loose = faults.inject(t, w["ta"], w["g"], "detached", start)
    assert np.allclose((loose - w["ta"])[lit], 0.6 * (t - w["ta"])[lit])


def test_cusum_finds_an_offset_quickly_and_stays_quiet_before(two_years):
    w, model, sensor, mask = two_years
    ref_period = w.index < pd.Timestamp("2022-01-01", tz="UTC")
    ref_feats = faults.daily_features(sensor[ref_period], model[ref_period], w["ta"][ref_period], mask[ref_period], TZ)
    ref = faults.reference_stats(ref_feats)
    h = faults.calibrate_h(ref_feats, ref)
    assert len(faults.cusum_alarms(ref_feats, ref, h)) <= 1

    start = pd.Timestamp("2022-05-01", tz="UTC")
    bad = faults.inject(sensor, w["ta"], w["g"], "offset_plus", start)
    test = ~ref_period
    feats = faults.daily_features(bad[test], model[test], w["ta"][test], mask[test], TZ)
    alarms = faults.cusum_alarms(feats, ref, h)
    d0 = start.tz_convert(TZ).normalize().tz_localize(None)
    delay = faults.first_alarm_after(alarms, d0, 60)
    assert np.isfinite(delay) and delay <= 10
    assert faults.label(feats, ref, d0 + pd.Timedelta(days=delay)) == "offset / drift up"


def test_cusum_labels_a_loose_sensor(two_years):
    w, model, sensor, mask = two_years
    ref_period = w.index < pd.Timestamp("2022-01-01", tz="UTC")
    ref_feats = faults.daily_features(sensor[ref_period], model[ref_period], w["ta"][ref_period], mask[ref_period], TZ)
    ref = faults.reference_stats(ref_feats)
    h = faults.calibrate_h(ref_feats, ref)
    start = pd.Timestamp("2022-03-01", tz="UTC")
    bad = faults.inject(sensor, w["ta"], w["g"], "detached", start, 0.6)
    test = ~ref_period
    feats = faults.daily_features(bad[test], model[test], w["ta"][test], mask[test], TZ)
    d0 = start.tz_convert(TZ).normalize().tz_localize(None)
    delay = faults.first_alarm_after(faults.cusum_alarms(feats, ref, h), d0, 60)
    assert np.isfinite(delay) and delay <= 10
    assert faults.label(feats, ref, d0 + pd.Timedelta(days=delay)) == "loss of contact"


def test_first_alarm_after_ignores_earlier_alarms():
    alarms = pd.DataFrame({"day": pd.to_datetime(["2022-01-05", "2022-02-10"]), "feature": "resid",
                           "direction": "up"})
    assert faults.first_alarm_after(alarms, pd.Timestamp("2022-02-01"), 60) == 9.0
    assert np.isnan(faults.first_alarm_after(alarms, pd.Timestamp("2022-03-01"), 60))
