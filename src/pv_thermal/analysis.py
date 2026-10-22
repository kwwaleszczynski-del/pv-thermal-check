"""The full study, section by section. Each function takes the prepared frame (prepare.build_frame) and returns
plain tables and numbers; run_all() chains them and writes results/.

Train/test conventions:
- module temperature models are fitted on 2017 and scored on 2016 (operating steps, POA > 50 W/m2);
- the sensor fault detector takes its reference statistics and threshold from 2017 and is tested on 2016;
- the inverter model's reference is the year after the July 2016 repair (2016-07-15 .. 2017-07-14), its injected
  faults are tested on the rest of 2017.

Usage:
    res = run_all(Path("data"), Path("results"), pvdaq_dir=None)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pv_thermal import faults, inverter, performance, poland, prepare, qc, thermal, transformer, voc
from pv_thermal.site import POLAND, POLAND_TILT, POLAND_YEARS, SYSTEM_34

log = logging.getLogger(__name__)

P = SYSTEM_34
MODELS = ("faiman", "sapm", "pvsyst", "ross")
LABELS = {"faiman": "Faiman", "sapm": "SAPM", "pvsyst": "PVsyst", "ross": "Ross"}
RATIOS = tuple(np.round(np.arange(0.5, 1.001, 0.05), 2))


@dataclass
class Results:
    summary: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)  # name -> DataFrame (written to results/)
    series: dict = field(default_factory=dict)  # name -> Series/DataFrame for figures (not written)


def _years(df: pd.DataFrame, *years: int) -> pd.Series:
    return pd.Series(df.index.year.isin(years), index=df.index)


# ---------------------------------------------------------------- module temperature models (on the sensor)
def section_models(df: pd.DataFrame, res: Results) -> dict:
    g, ta, ws, y = df["g"], df["ta"], df["ws"], df["t_mod"]
    base = df["operating"] & (g > 50) & y.notna() & ta.notna() & ws.notna()
    tr, te = base & _years(df, 2017), base & _years(df, 2016)
    rows, preds = [], {}
    for m in MODELS:
        eff = P.module_efficiency
        default = thermal.predict(m, thermal.DEFAULTS[m], g, ta, ws, efficiency=eff)
        st = thermal.fit_model(m, g, ta, ws, y, tr, efficiency=eff)
        dy = thermal.fit_model(m, g, ta, ws, y, tr, dynamic=True, efficiency=eff)
        p_st = thermal.predict(m, st.params, g, ta, ws, efficiency=eff)
        p_dy = thermal.predict(m, dy.params, g, ta, ws, efficiency=eff)
        p_pr = thermal.prilliman(p_st, ws)
        for variant, pred, params in [("default", default, thermal.DEFAULTS[m]), ("fitted", p_st, st.params),
                                      ("fitted + lag", p_dy, dy.params), ("fitted + Prilliman", p_pr, st.params)]:
            e = thermal.errors(pred, y, te)
            rows.append({"model": LABELS[m], "variant": variant, "rmse": e["rmse"], "bias": e["bias"],
                         "mae": e["mae"], "n_test": e["n"],
                         "params": json.dumps({k: round(v, 4) for k, v in params.items()})})
        preds[m] = {"static": p_st, "lag": p_dy, "prilliman": p_pr, "default": default, "fit": st, "fit_lag": dy}
    table = pd.DataFrame(rows)
    res.tables["module_models"] = table
    fa = preds["faiman"]
    hours, months = {}, {}
    for key in ("default", "static", "lag", "prilliman"):
        hours[key], months[key] = thermal.bias_table(fa[key], y, te, P.tz)
    res.tables["bias_by_hour"] = pd.DataFrame(hours).rename_axis("local_hour")
    res.tables["bias_by_month"] = pd.DataFrame(months).rename_axis("month")
    res.series["faiman_lag"] = fa["lag"]
    res.series["faiman_static"] = fa["static"]
    res.series["faiman_default"] = fa["default"]
    best = table[table["variant"] != "default"].sort_values("rmse").iloc[0]
    out = {
        "train": "2017", "test": "2016", "best": f"{best['model']} {best['variant']}", "best_rmse": float(best["rmse"]),
        "faiman_fit": fa["fit"].params, "faiman_lag_fit": fa["fit_lag"].params,
        "rmse_default": {LABELS[m]: float(table.query("model == @LABELS[@m] and variant == 'default'")["rmse"].iloc[0])
                         for m in MODELS},
        "rmse_fitted": {LABELS[m]: float(table.query("model == @LABELS[@m] and variant == 'fitted'")["rmse"].iloc[0])
                        for m in MODELS},
        "bias_default_faiman": float(table.query("model == 'Faiman' and variant == 'default'")["bias"].iloc[0]),
    }
    log.info("models: best %s, rmse %.2f K", out["best"], out["best_rmse"])
    return out


# ---------------------------------------------------------------- Voc thermometry: is the sensor right?
def section_voc(df: pd.DataFrame, res: Results) -> dict:
    oc = df[df["open_circuit"]]
    v = voc.voc_temperature(oc, P, df["g"])
    v["ta"], v["ws"] = df["ta"], df["ws"]
    bins = [20, 200, 400, 600, 800, 1000, 1400]
    v["g_bin"] = pd.cut(v["g"], bins)
    binned = v.groupby("g_bin", observed=True).apply(lambda x: pd.Series({
        "n": len(x), "cell_minus_sensor": (x["t_cell_voc"] - x["t_mod"]).median(),
        "sensor_rise": (x["t_mod"] - x["ta"]).median(), "cell_rise": (x["t_cell_voc"] - x["ta"]).median()}))
    binned.index = binned.index.astype(str)
    res.tables["voc_thermometry"] = binned
    h = v[(v["g"] > 400)].dropna(subset=["t_cell_voc", "t_mod", "ta"])
    # share of the cell's rise over air that the back-of-module sensor reports (least squares through 0),
    # after removing the expected 3 K/1000 W/m2 cell-to-back difference
    back = h["t_cell_voc"] - h["g"] / 1000 * thermal.SAPM_DT - h["ta"]
    gain = float(((h["t_mod"] - h["ta"]) * back).sum() / (back ** 2).sum())
    target = pd.Series(np.nan, index=df.index)
    target[v.index] = v["t_cell_voc"]
    mask = target.notna() & (df["g"] > 100)
    pv_oc = thermal.fit_model("pvsyst", df["g"], df["ta"], df["ws"], target, mask, efficiency=0.0)
    fa_oc = thermal.fit_model("faiman", df["g"], df["ta"], df["ws"], target, mask)
    lag = thermal.fit_model("pvsyst", df["g"], df["ta"], df["ws"], target, mask, dynamic=True, efficiency=0.0)
    # at the maximum power point the module turns part of the absorbed light into electricity: Faiman's lumped
    # U-values scale by alpha / (alpha - efficiency); PVsyst carries the efficiency explicitly
    scale = thermal.ALPHA / (thermal.ALPHA - P.module_efficiency)
    res.series["voc"] = v
    # sensitivity: the CEC single-diode model's own (steeper) Voc temperature dependence
    vc = voc.voc_temperature(oc, P, df["g"], method="cec")
    hc = vc.join(df[["ta"]])[vc["g"] > 800].dropna()
    target_c = pd.Series(np.nan, index=df.index)
    target_c[vc.index] = vc["t_cell_voc"]
    pv_cec = thermal.fit_model("pvsyst", df["g"], df["ta"], df["ws"], target_c, target_c.notna() & (df["g"] > 100),
                               efficiency=0.0)
    out = {
        "open_circuit_steps": int(len(v)), "open_circuit_days": int(len(set(v.index.tz_convert(P.tz).date))),
        "calibration_scale": v.attrs["scale"], "calibration_steps": v.attrs["n_calib"],
        "cell_minus_sensor_g800": float((h["t_cell_voc"] - h["t_mod"])[h["g"] > 800].median()),
        "cell_rise_g800": float((h["t_cell_voc"] - h["ta"])[h["g"] > 800].median()),
        "sensor_rise_g800": float((h["t_mod"] - h["ta"])[h["g"] > 800].median()),
        "sensor_gain": gain,
        "pvsyst_oc": pv_oc.params, "pvsyst_oc_rmse": pv_oc.rmse, "pvsyst_oc_tau_min": lag.params["tau"],
        "faiman_oc": fa_oc.params,
        "beta_voc_v_per_k": voc.datasheet_beta_voc(P.module),
        "cec_sensitivity": {"cell_minus_sensor_g800": float((hc["t_cell_voc"] - hc["t_mod"]).median()),
                            "cell_rise_g800": float((hc["t_cell_voc"] - hc["ta"]).median()),
                            "pvsyst_oc": pv_cec.params, "calibration_scale": vc.attrs["scale"]},
        "faiman_mpp": {"u0": fa_oc.params["u0"] * scale, "u1": fa_oc.params["u1"] * scale},
    }
    log.info("Voc: cells %.1f K above the sensor at > 800 W/m2, sensor reports %.0f %% of the rise",
             out["cell_minus_sensor_g800"], 100 * gain)
    return out


def best_cell_temperature(df: pd.DataFrame, voc_out: dict) -> pd.Series:
    """Best estimate of cell temperature: PVsyst with the Voc-calibrated U-values, electrical efficiency while
    the inverter runs and zero while the array sits at open circuit."""
    p = voc_out["pvsyst_oc"]
    run = thermal.predict("pvsyst", p, df["g"], df["ta"], df["ws"], efficiency=P.module_efficiency)
    idle = thermal.predict("pvsyst", p, df["g"], df["ta"], df["ws"], efficiency=0.0)
    return run.where(df["operating"], idle)


# ---------------------------------------------------------------- temperature losses and PR
def section_performance(df: pd.DataFrame, t_cell: pd.Series, res: Results) -> dict:
    df, t_cell = df.tz_convert(P.tz), t_cell.tz_convert(P.tz)  # months in the plant's local time
    g = df["g"]
    t_sensor = performance.cell_from_back(df["t_mod"], g)
    loss = pd.DataFrame({"model_cell": performance.temperature_loss(g, t_cell, P.gamma_pdc),
                         "sensor_plus_3k": performance.temperature_loss(g, t_sensor, P.gamma_pdc)})
    res.tables["temperature_loss_monthly"] = loss
    elev = qc.clearsky_poa(df.index, P)["elevation"]
    mask = df["operating"] & (g > 50) & (elev > 30) & t_cell.notna() & t_sensor.notna()
    pr_model = performance.monthly_pr(df["p_dc"], g, t_cell, P.pdc0_kw, P.gamma_pdc, mask)
    pr_sensor = performance.monthly_pr(df["p_dc"], g, t_sensor, P.pdc0_kw, P.gamma_pdc, mask)
    pr = pr_model.join(pr_sensor[["pr_stc", "pr_nrel", "t_cell_w"]], rsuffix="_sensor")
    res.tables["pr_monthly"] = pr
    seas = {k: performance.seasonality(pr[k], pr["t_cell_w"]) for k in ("pr_raw", "pr_stc", "pr_nrel")}
    seas_sensor = {k: performance.seasonality(pr[f"{k}_sensor"], pr["t_cell_w"]) for k in ("pr_stc", "pr_nrel")}
    mask_all = df["operating"] & (g > 50) & t_cell.notna()
    pr_all = performance.monthly_pr(df["p_dc"], g, t_cell, P.pdc0_kw, P.gamma_pdc, mask_all)
    out = {
        "annual_loss_model": {y: performance.temperature_loss(g[_years(df, y)], t_cell[_years(df, y)], P.gamma_pdc,
                                                              None) for y in (2016, 2017)},
        "annual_loss_sensor": {y: performance.temperature_loss(g[_years(df, y)], t_sensor[_years(df, y)],
                                                               P.gamma_pdc, None) for y in (2016, 2017)},
        "worst_month_loss": float(loss["model_cell"].min()),
        "t_typ_model": pr_model.attrs["t_typ"], "t_typ_sensor": pr_sensor.attrs["t_typ"],
        "pr_filter": "operating, POA > 50 W/m2, sun elevation > 30 deg, DC side",
        "seasonality": seas, "seasonality_sensor": seas_sensor,
        "seasonality_all_elevations": {k: performance.seasonality(pr_all[k], pr_all["t_cell_w"])
                                       for k in ("pr_raw", "pr_stc", "pr_nrel")},
    }
    log.info("losses: %.1f %% / yr (cell model) vs %.1f %% (sensor); PR spread %.3f -> %.3f (NREL)",
             100 * out["annual_loss_model"][2017], 100 * out["annual_loss_sensor"][2017], seas["pr_raw"]["spread"],
             seas["pr_nrel"]["spread"])
    return out


# ---------------------------------------------------------------- sensor fault detection
def section_faults(df: pd.DataFrame, t_cell: pd.Series, res: Results, step_days: int = 7, horizon: int = 60) -> dict:
    g, ta, ws, y = df["g"], df["ta"], df["ws"], df["t_mod"]
    base = df["operating"] & (g > 50) & y.notna() & ta.notna() & ws.notna()
    fit = thermal.fit_model("faiman", g, ta, ws, y, base & _years(df, 2017), dynamic=True)
    pred = thermal.predict("faiman", fit.params, g, ta, ws)
    mask = df["operating"] & (g > 200)
    feats = faults.daily_features(y, pred, ta, mask, P.tz)
    f17, f16 = feats[feats.index.year == 2017], feats[feats.index.year == 2016]
    ref = faults.reference_stats(f17)
    h = faults.calibrate_h(f17, ref)
    clean = faults.cusum_alarms(f16, ref, h)
    clean_shew = faults.shewhart_alarms(f16, ref)
    starts = pd.date_range("2016-01-15", "2016-10-15", freq=f"{step_days}D", tz="UTC")
    cases = [(k, None) for k in faults.FAULTS] + [("offset_plus", s) for s in (1.0, 2.0, 5.0)] + \
            [("detached", s) for s in (0.9, 0.8)]
    rows = []
    for kind, size in cases:
        t_c, t_s, labels = [], [], []
        for st in starts:
            yf = faults.inject(y, ta, g, kind, st, size)
            ff = faults.daily_features(yf, pred, ta, mask, P.tz)
            ff = ff[ff.index.year == 2016]
            d0 = st.tz_convert(P.tz).normalize().tz_localize(None)
            t = faults.first_alarm_after(faults.cusum_alarms(ff, ref, h), d0, horizon)
            t_c.append(t)
            if np.isfinite(t):
                labels.append(faults.label(ff, ref, d0 + pd.Timedelta(days=t)))
            t_s.append(faults.first_alarm_after(faults.shewhart_alarms(ff, ref), d0, horizon))
        t_c, t_s = np.array(t_c), np.array(t_s)
        expected = {"offset_plus": "offset / drift up", "offset_minus": "offset / drift down",
                    "drift": "offset / drift up", "detached": "loss of contact",
                    "ventilation": "reduced ventilation"}[kind]
        rows.append({"fault": kind, "size": size if size is not None else {"offset_plus": 3.0, "offset_minus": 3.0,
                     "drift": 4.5, "detached": 0.6, "ventilation": 1.3}[kind], "default_size": size is None,
                     "trials": len(starts), "cusum_detected": float(np.isfinite(t_c).mean()),
                     "cusum_median_days": float(np.nanmedian(t_c)) if np.isfinite(t_c).any() else np.nan,
                     "cusum_p90_days": float(np.nanpercentile(t_c, 90)) if np.isfinite(t_c).any() else np.nan,
                     "label_correct": float(np.mean([lab == expected for lab in labels])) if labels else np.nan,
                     "shewhart_detected": float(np.isfinite(t_s).mean()),
                     "shewhart_median_days": float(np.nanmedian(t_s)) if np.isfinite(t_s).any() else np.nan})
    table = pd.DataFrame(rows)
    res.tables["sensor_faults"] = table
    # the real sensor against physics instead of its own history: expected back-of-module temperature from the
    # Voc-calibrated cell model
    expected_back = t_cell - g / 1000 * thermal.SAPM_DT
    phys = faults.daily_features(y, expected_back, ta, mask, P.tz)
    res.series["features_self"] = feats
    res.series["features_physics"] = phys
    # one example for the figure: a sensor coming loose on 1 March 2016
    ex_start = pd.Timestamp("2016-03-01", tz="UTC")
    ex = faults.daily_features(faults.inject(y, ta, g, "detached", ex_start), pred, ta, mask, P.tz)
    res.series["example_detached"] = ex[ex.index.year == 2016]
    res.series["example_alarms"] = faults.cusum_alarms(ex[ex.index.year == 2016], ref, h)
    out = {
        "reference": "2017", "test": "2016", "threshold_h": h, "k": 0.5,
        "ref_mean": ref.mean.to_dict(), "ref_std": ref.std.to_dict(),
        "clean_alarms_2016": [str(d.date()) for d in clean["day"]],
        "clean_alarms_shewhart_2016": int(len(clean_shew)),
        "starts": len(starts), "horizon_days": horizon,
        "physics_ratio_median": float(phys["ratio"].median()),
        "physics_ratio_iqr": [float(phys["ratio"].quantile(0.25)), float(phys["ratio"].quantile(0.75))],
        "physics_days": int(len(phys)),
    }
    log.info("faults: h=%.1f, %d clean alarms in 2016; physics ratio %.2f", h, len(clean), out["physics_ratio_median"])
    return out


# ---------------------------------------------------------------- inverter
def section_inverter(df: pd.DataFrame, t_cell: pd.Series, res: Results) -> dict:
    warm = inverter.warm_mask(df)
    ref_period = (df.index >= pd.Timestamp("2016-07-15", tz="UTC")) & (df.index < pd.Timestamp("2017-07-15", tz="UTC"))
    coef = inverter.fit_inverter(df, warm & ref_period, P.pac0_kw)
    pred = inverter.predict_inverter(df, coef, P.pac0_kw)
    daily = inverter.daily_residual(df, pred, warm, P.tz)
    ref_days = daily[(daily.index >= "2016-07-15") & (daily.index < "2017-07-15")]
    med = ref_days["resid"].median()
    ref = faults.Reference(pd.Series({"resid": med}), pd.Series({"resid": (ref_days["resid"] - med).abs().median()
                                                                   * 1.4826}))
    h = faults.calibrate_h(ref_days, ref, cols=("resid",))
    real = faults.cusum_alarms(daily, ref, h, cols=("resid",))
    res.series["inverter_daily"] = daily
    res.series["inverter_alarms"] = real
    monthly = (df["t_inv"] - pred)[warm].groupby(df.index[warm].tz_convert(P.tz).tz_localize(None).to_period("M")
                                                 ).mean()
    res.tables["inverter_residual_monthly"] = monthly.rename("resid").to_frame()
    pre = daily[daily.index < "2016-06-01"]["resid"].mean()
    post = daily[daily.index >= "2016-07-15"]["resid"].mean()
    # injected cooling faults on the test period after the reference year
    test_start = pd.Timestamp("2017-07-15")
    starts = pd.date_range("2017-07-17", "2017-10-30", freq="7D")
    rows = []
    for kind in ("fan", "offset"):
        t_det = []
        for st in starts:
            d2 = df.copy()
            on = (d2.index.tz_convert(P.tz).tz_localize(None) >= st)
            if kind == "fan":  # heat removal reduced: extra rise proportional to the heat load
                d2.loc[on, "t_inv"] = d2.loc[on, "t_inv"] + 4.0 * d2.loc[on, "p_ac"].clip(lower=0) / P.pac0_kw
            else:
                d2.loc[on, "t_inv"] = d2.loc[on, "t_inv"] + 1.5
            dd = inverter.daily_residual(d2, pred, warm, P.tz)
            dd = dd[dd.index >= test_start]
            t_det.append(faults.first_alarm_after(faults.cusum_alarms(dd, ref, h, cols=("resid",)), st, 60))
        t_det = np.array(t_det)
        rows.append({"fault": {"fan": "fan: +4 K at rated load", "offset": "step +1.5 K"}[kind],
                     "trials": len(starts), "detected": float(np.isfinite(t_det).mean()),
                     "median_days": float(np.nanmedian(t_det)) if np.isfinite(t_det).any() else np.nan})
    clean_test = faults.cusum_alarms(daily[daily.index >= test_start], ref, h, cols=("resid",))
    # derating
    exp_ac, eff = inverter.expected_ac(df, t_cell, P)
    lit = df["operating"] & (df["g"] > 300)
    flags = inverter.derating_flags(df["p_ac"], exp_ac, lit, P.pac0_kw)
    cap, t_above, thr = 90.0, 35.0, 0.08
    inj = inverter.inject_derating(df, cap_kw=cap, t_air_above=t_above)
    exp_inj, _ = inverter.expected_ac(inj, t_cell, P)
    f_inj = inverter.derating_flags(inj["p_ac"], exp_inj, lit, P.pac0_kw, shortfall=thr)
    capped = inj.attrs["derated_steps"] & lit
    truth = capped & (df["p_ac"] > cap / (1 - thr))  # the cap removed more than the detection threshold
    bins = [-10, 10, 20, 30, 35, 40, 45, 50]
    high = lit & (exp_ac >= 0.6 * P.pac0_kw)  # where a power cap would act
    by_ta = pd.DataFrame({
        "real": (1 - df["p_ac"] / exp_ac).where(high), "injected": (1 - inj["p_ac"] / exp_inj).where(high),
        "real_flag": flags.astype(float).where(high), "injected_flag": f_inj.astype(float).where(high),
        "ta": df["ta"]}).dropna()
    by_ta = by_ta.groupby(pd.cut(by_ta["ta"], bins), observed=True).agg(
        real_median_shortfall=("real", "median"), injected_median_shortfall=("injected", "median"),
        real_flagged_share=("real_flag", "mean"), injected_flagged_share=("injected_flag", "mean"),
        steps=("real", "size"))
    by_ta.index = by_ta.index.astype(str)
    res.tables["derating_by_air_temperature"] = by_ta
    days = df.index.tz_convert(P.tz).normalize()
    day_truth = truth.groupby(days).sum() >= 2
    day_capped = capped.groupby(days).sum() >= 1
    day_flag = f_inj.groupby(days).sum() >= 2
    day_flag_real = flags.groupby(days).sum() >= 2
    out = {
        "coef": coef, "reference": "2016-07-15 .. 2017-07-14",
        "t_inv_operating_range": [float(df.loc[warm, "t_inv"].quantile(0.01)),
                                  float(df.loc[warm, "t_inv"].quantile(0.99))],
        "residual_before_jun2016": float(pre), "residual_after_jul2016": float(post),
        "threshold_h": h,
        "real_alarms": {dirn: {"n": int(len(g)), "first": str(g["day"].min().date()), "last": str(g["day"].max().date())}
                        for dirn, g in real.groupby("direction")},
        "clean_test_alarms": int(len(clean_test)), "injected": rows,
        "conversion_efficiency": eff, "derating_flagged_steps": int(flags.sum()), "lit_steps": int(lit.sum()),
        "derating_flagged_days": int(day_flag_real.sum()),
        "conversion_efficiency_by_air": _by_bins((df["p_ac"] / df["p_dc"]).where(df["operating"] & (df["p_dc"] > 20)),
                                                df["ta"], [-10, 20, 30, 40, 50]),
        "shortfall_by_inverter_temp": {str(int(k)): float(v) for k, v in (1 - df["p_ac"] / exp_ac).where(lit)
                                       .groupby(df["t_inv"]).median().loc[54:].items()},
        "shortfall_am_pm": {half: _by_bins((1 - df["p_ac"] / exp_ac).where(lit & sel), df["ta"], [-10, 20, 35, 50])
                            for half, sel in (("am", df.index.tz_convert(P.tz).hour < 12),
                                              ("pm", df.index.tz_convert(P.tz).hour >= 12))},
        "derating_injection": f"AC capped at {cap:.0f} kW whenever air > {t_above:.0f} degC",
        "derating_injected_true_steps": int(truth.sum()), "derating_injected_true_days": int(day_truth.sum()),
        "derating_step_recall": float((f_inj & truth).sum() / max(truth.sum(), 1)),
        "derating_step_precision": float((f_inj & capped).sum() / max(f_inj.sum(), 1)),
        "derating_day_recall": float((day_flag & day_truth).sum() / max(day_truth.sum(), 1)),
        "derating_day_precision": float((day_flag & day_capped).sum() / max(day_flag.sum(), 1)),
        "derating_injected_flagged": int(f_inj.sum()),
        "max_air_temp_operating": float(df.loc[lit, "ta"].max()),
    }
    res.tables["inverter_injected"] = pd.DataFrame(rows)
    log.info("inverter: T = %.1f + %.2f P/Pr + %.3f Ta; derating flags %d steps; injected day recall %.2f "
             "precision %.2f", coef["c0"], coef["c_p"], coef["c_a"], out["derating_flagged_steps"],
             out["derating_day_recall"], out["derating_day_precision"])
    return out


# ---------------------------------------------------------------- Poland and the transformer
def section_poland(df: pd.DataFrame, models_out: dict, voc_out: dict, data_dir: Path, res: Results) -> dict:
    start, end = f"{POLAND_YEARS[0]}-01-01", f"{POLAND_YEARS[1]}-12-31"
    weather = poland.fetch_era5(POLAND, start, end, Path(data_dir) / "era5")
    cand = {
        "Voc-calibrated (this plant)": ("pvsyst", voc_out["pvsyst_oc"], P.module_efficiency, True),
        "Fitted to the faulty sensor": ("faiman", models_out["faiman_fit"], 0.0, False),
        "Faiman default": ("faiman", thermal.DEFAULTS["faiman"], 0.0, True),
        "SAPM open rack": ("sapm", thermal.DEFAULTS["sapm"], 0.0, True),
        "PVsyst free-standing": ("pvsyst", thermal.DEFAULTS["pvsyst"], P.module_efficiency, True),
        "Ross / NOCT": ("ross", thermal.DEFAULTS["ross"], 0.0, True),
    }
    rows, sizing, hot_week = [], [], {}
    sites = {"Las Vegas (measured POA)": None, **POLAND}
    for site, ll in sites.items():
        if ll is None:
            yr = _years(df, 2016, 2017)
            g, ta, ws = df["g"][yr], df["ta"][yr].interpolate(limit_direction="both"), \
                df["ws"][yr].interpolate(limit_direction="both")
        else:
            w = weather[site]
            g, ta, ws = poland.plane_of_array(w, *ll, POLAND_TILT), w["t_air"], w["wind"]
        row = {"site": site, "poa_kwh_m2_yr": float(g.sum() * _step_h(g) / 1000 / _n_years(g)),
               "mean_air_c": float(ta.mean())}
        for name, (m, p, eff, cell) in cand.items():
            tc = thermal.predict(m, p, g, ta, ws, efficiency=eff, cell=cell)
            if m == "faiman" and not cell:  # fitted to a back-of-module sensor: add the cell offset
                tc = tc + g / 1000 * thermal.SAPM_DT
            row[name] = performance.temperature_loss(g, tc, P.gamma_pdc, None)
        rows.append(row)
        tc = thermal.predict("pvsyst", voc_out["pvsyst_oc"], g, ta, ws, efficiency=P.module_efficiency)
        pac = poland.ac_power(g, tc, 1000.0, P.gamma_pdc)
        tab = transformer.sizing_table(pac, ta, 1000.0 / 1.2, RATIOS)
        tab.insert(0, "site", site)
        sizing.append(tab)
        if site in ("Warsaw", "Las Vegas (measured POA)"):
            hot_week[site] = (pac, ta)
    losses = pd.DataFrame(rows).set_index("site")
    sizing = pd.concat(sizing, ignore_index=True)
    res.tables["temperature_loss_by_site"] = losses
    res.tables["transformer_sizing"] = sizing
    res.series["hot_week"] = hot_week
    model_cols = list(cand)
    pl = losses.drop(index="Las Vegas (measured POA)")
    smallest = {s: transformer.smallest_ratio(sizing[sizing["site"] == s]) for s in sites}
    spread = pl[[c for c in model_cols if c != "Fitted to the faulty sensor"]]
    out = {
        "years": f"{POLAND_YEARS[0]}-{POLAND_YEARS[1]}", "tilt": POLAND_TILT, "sites": list(POLAND),
        "loss_best_range": [float(pl[model_cols[0]].min()), float(pl[model_cols[0]].max())],
        "loss_best_las_vegas": float(losses.loc["Las Vegas (measured POA)", model_cols[0]]),
        "model_spread_pp": float(100 * (spread.max(axis=1) - spread.min(axis=1)).mean()),
        "loss_sensor_fit_range": [float(pl[model_cols[1]].min()), float(pl[model_cols[1]].max())],
        "transformer_smallest_ratio": smallest,
        "transformer_ageing_at_0_8": {s: float(sizing[(sizing["site"] == s) & np.isclose(sizing["ratio"], 0.8)]
                                               ["ageing_days_per_year"].iloc[0]) for s in sites},
    }
    log.info("Poland: best-estimate loss %.1f..%.1f %%, model spread %.1f pp; transformer ratios %s",
             100 * out["loss_best_range"][1], 100 * out["loss_best_range"][0], out["model_spread_pp"], smallest)
    return out


def _by_bins(values: pd.Series, key: pd.Series, bins: list) -> dict:
    d = pd.DataFrame({"v": values, "k": key}).dropna()
    m = d.groupby(pd.cut(d["k"], bins), observed=True)["v"].median()
    return {str(i): float(v) for i, v in m.items()}


def _step_h(s: pd.Series) -> float:
    return (s.index[1] - s.index[0]).total_seconds() / 3600


def _n_years(s: pd.Series) -> float:
    return len(s) * _step_h(s) / 8760


# ---------------------------------------------------------------- driver
def run_all(data_dir: Path, results_dir: Path, pvdaq_dir: Path | None = None) -> Results:
    df, qc_info = prepare.build_frame(data_dir, pvdaq_dir)
    res = Results()
    res.series["frame"] = df
    res.summary["qc"] = qc_info
    res.summary["models"] = section_models(df, res)
    res.summary["voc"] = section_voc(df, res)
    t_cell = best_cell_temperature(df, res.summary["voc"])
    res.series["t_cell"] = t_cell
    res.summary["performance"] = section_performance(df, t_cell, res)
    res.summary["faults"] = section_faults(df, t_cell, res)
    res.summary["inverter"] = section_inverter(df, t_cell, res)
    res.summary["poland"] = section_poland(df, res.summary["models"], res.summary["voc"], data_dir, res)
    write(res, results_dir)
    return res


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write(res: Results, results_dir: Path) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(_clean(res.summary), indent=1), encoding="utf-8")
    for name, t in res.tables.items():
        t.round(5).to_csv(out / f"{name}.csv")
    log.info("wrote %s (%d tables)", out, len(res.tables))
