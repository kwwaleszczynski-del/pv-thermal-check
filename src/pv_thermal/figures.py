"""Static PNG figures for the README (matplotlib, light theme).

Colour roles, the same in every figure:
    blue   = cell temperature from physics (Voc thermometry, the Voc-calibrated model), corrected quantities
    orange = the back-of-module sensor and anything fitted to it
    aqua   = published default coefficients, injected scenarios
    grey   = air temperature, raw values, baselines

Usage:
    paths = save_all(res, Path("docs/img"))
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pv_thermal.site import SYSTEM_34  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CELL, SENSOR, DEFAULT = "#2a78d6", "#eb6834", "#1baf7a"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = ["#86b6ef", "#2a78d6", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})
TZ = SYSTEM_34.tz


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def fig_site_sensors(res, path: Path) -> Path:
    s, df = res.summary["qc"], res.series["frame"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    st = pd.Series(s["ambient"]["stuck_share_by_month"]).sort_index()
    st.index = st.index.astype(int)
    a1.bar(st.index, 100 * st.to_numpy(), width=0.7, color=SENSOR)
    a1.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a1.set_ylabel("% of readings")
    a1.set_title("Ambient sensor stuck at 0 degF")
    h = pd.DataFrame({"site": df["ws_site"], "station": df["ws"]}).resample("1h").mean().dropna()
    a2.hexbin(h["station"], h["site"], gridsize=30, cmap="Blues", mincnt=1, linewidths=0, extent=(0, 14, 0, 6),
              bins="log")
    a2.plot([0, 6], [0, 6], color=AXIS, lw=1)
    a2.set_xlabel("airport anemometer, m/s (10 m)")
    a2.set_ylabel("site anemometer, m/s")
    a2.set_title(f"Wind: site {s['wind']['site_mean_ms']:.1f} vs airport {s['wind']['station_mean_ms']:.1f} m/s")
    r = pd.Series(s["poa_ratio_by_month"]).sort_index()
    r.index = r.index.astype(int)
    a3.axhline(1, color=AXIS, lw=1)
    a3.plot(r.index, r.to_numpy(), color=SENSOR, marker="o", ms=6)
    a3.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a3.set_ylim(0.9, 1.3)
    a3.set_ylabel("POA sensor / clear-sky model")
    a3.set_title(f"POA sensor reads {100 * (s['poa_scale'] - 1):.0f} % high on clear days")
    fig.tight_layout()
    return _save(fig, path)


def fig_models(res, path: Path) -> Path:
    t = res.tables["module_models"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    models = list(dict.fromkeys(t["model"]))
    x = np.arange(len(models))
    w = 0.36
    for i, (variant, colour, label) in enumerate([("default", DEFAULT, "published defaults"),
                                                 ("fitted + lag", SENSOR, "fitted to the sensor + lag")]):
        v = [float(t.query("model == @m and variant == @variant")["rmse"].iloc[0]) for m in models]
        bars = a1.bar(x + (i - 0.5) * w, v, w - 0.04, color=colour, label=label)
        a1.bar_label(bars, fmt="%.1f", fontsize=8, color=INK2, padding=2)
    a1.set_xticks(x, models)
    a1.set_ylabel("RMSE on 2016, K")
    a1.set_title("Module-sensor temperature: test error")
    a1.legend(loc="upper left")
    b = res.tables["bias_by_hour"]
    a2.axhline(0, color=AXIS, lw=1)
    a2.plot(b.index, b["static"], color=SENSOR, marker="o", ms=5, label="Faiman fitted, steady state")
    a2.plot(b.index, b["lag"], color=INK2, marker="o", ms=5, label="Faiman fitted + lag")
    a2.set_xlabel("local standard time, h")
    a2.set_ylabel("mean error (model - sensor), K")
    a2.set_title("Bias by time of day (2016)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def fig_voc(res, path: Path) -> Path:
    v = res.series["voc"]
    s = res.summary["voc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    df = res.series["frame"].loc["2016-06-06 12:00":"2016-06-10 06:00"]
    loc = df.index.tz_convert(TZ).tz_localize(None)
    cell = v["t_cell_voc"].reindex(df.index)  # NaN at night: no open-circuit voltage without light
    a1.plot(loc, cell, color=CELL, marker="o", ms=3, lw=1.5, label="cell, from open-circuit voltage")
    a1.plot(loc, df["t_mod"], color=SENSOR, lw=1.5, label="back-of-module sensor")
    a1.plot(loc, df["ta"], color=INK2, lw=1.5, label="air (airport)")
    a1.set_ylabel("degC")
    a1.xaxis.set_major_locator(mdates.DayLocator())  # one tick per day (the default put two per day, same label)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a1.set_title("Inverter down, array at open circuit: June 2016")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    h = v.dropna(subset=["t_cell_voc", "t_mod", "ta"])
    h = h[h["g"] > 20]
    a2.scatter(h["g"], h["t_cell_voc"] - h["ta"], s=10, color=CELL, alpha=0.6, linewidths=0, label="cell (Voc)")
    a2.scatter(h["g"], h["t_mod"] - h["ta"], s=10, color=SENSOR, alpha=0.6, linewidths=0, label="sensor")
    a2.set_xlabel("POA irradiance, W/m2 (clear-sky scale)")
    a2.set_ylabel("rise over air, K")
    a2.set_title(f"Sensor reports {100 * s['sensor_gain']:.0f} % of the cell's rise")
    a2.legend(loc="lower right")  # empty corner: high irradiance, no rise below air
    fig.tight_layout()
    return _save(fig, path)


def fig_losses_pr(res, path: Path) -> Path:
    loss, pr = res.tables["temperature_loss_monthly"], res.tables["pr_monthly"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9))
    x = loss.index.tz_convert(None) if loss.index.tz is not None else loss.index
    a1.axhline(0, color=AXIS, lw=1)
    a1.plot(x, 100 * loss["model_cell"], color=CELL, marker="o", ms=5, label="cell temperature (Voc-calibrated)")
    a1.plot(x, 100 * loss["sensor_plus_3k"], color=SENSOR, marker="o", ms=5, label="sensor + 3 K")
    a1.set_ylabel("loss vs 25 degC, %")
    a1.set_title("Monthly temperature loss")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    xp = pr.index.tz_convert(None) if pr.index.tz is not None else pr.index
    a2.plot(xp, pr["pr_raw"], color=INK2, marker="o", ms=5, label="raw PR (IEC 61724-1)")
    a2.plot(xp, pr["pr_stc"], color=DEFAULT, marker="o", ms=5, label="corrected to 25 degC")
    a2.plot(xp, pr["pr_nrel"], color=CELL, marker="o", ms=5, label="NREL weather-corrected")
    s = res.summary["performance"]["seasonality"]
    a2.set_title(f"Monthly DC PR: spread {100 * s['pr_raw']['spread']:.1f} -> {100 * s['pr_nrel']['spread']:.1f} pp")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a2.set_ylabel("PR (sun above 30 deg)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def fig_faults(res, path: Path) -> Path:
    fs, fp = res.series["features_self"], res.series["features_physics"]
    ex, al = res.series["example_detached"], res.series["example_alarms"]
    t = res.tables["sensor_faults"]
    t = t[t["default_size"]]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    a1.axhline(1, color=AXIS, lw=1)
    a1.plot(fs.index, fs["ratio"].rolling(7, center=True).median(), color=SENSOR, lw=1.5,
            label="vs model fitted to the sensor")
    a1.plot(fp.index, fp["ratio"].rolling(7, center=True).median(), color=CELL, lw=1.5,
            label="vs physics (Voc-calibrated)")
    a1.set_ylim(0, 1.6)
    a1.set_ylabel("daily rise ratio (7-day median)")
    a1.set_title("The real sensor: fine against itself, not physics")
    a1.legend(loc="lower left", fontsize=9)
    a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a2.axhline(1, color=AXIS, lw=1)
    a2.plot(ex.index, ex["ratio"], color=INK2, lw=1, marker="o", ms=3, label="daily rise ratio")
    a2.axvline(pd.Timestamp("2016-03-01"), color=DEFAULT, lw=1.5, label="sensor comes loose (injected)")
    if len(al):
        first = al[al["day"] >= "2016-03-01"]["day"].iloc[0]
        a2.axvline(first, color=SENSOR, lw=1.5, ls="--", label=f"CUSUM alarm ({(first - pd.Timestamp('2016-03-01')).days} d)")
    a2.set_xlim(pd.Timestamp("2016-01-15"), pd.Timestamp("2016-04-15"))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a2.set_title("Injected fault: rise x 0.6 from 1 March 2016")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)  # below the axis, off the data
    names = {"offset_plus": "+3 K", "offset_minus": "-3 K", "drift": "drift 4.5 K/90 d", "detached": "loose x0.6",
             "ventilation": "ventilation x1.3"}
    y = np.arange(len(t))
    a3.barh(y - 0.18, t["cusum_median_days"], 0.34, color=CELL, label="CUSUM (offset + ratio)")
    a3.barh(y + 0.18, t["shewhart_median_days"], 0.34, color=AXIS, label="3-sigma on daily residual")
    for yi, (d, sd) in enumerate(zip(t["cusum_detected"], t["shewhart_detected"])):
        a3.text(max(t["cusum_median_days"].max(), t["shewhart_median_days"].max()) * 1.05, yi,
                f"{100 * d:.0f}% / {100 * sd:.0f}%", va="center", fontsize=8, color=INK2)
    a3.set_yticks(y, [names[f] for f in t["fault"]])
    a3.invert_yaxis()
    a3.set_xlabel("median days to alarm\n(right: share found within 60 d, CUSUM / baseline)")
    a3.set_title("Detection on 40 start dates (2016)")
    a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1, fontsize=8)  # below the two-line label
    fig.tight_layout()
    return _save(fig, path)


def fig_inverter(res, path: Path) -> Path:
    df = res.series["frame"]
    s = res.summary["inverter"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
    m = df["operating"] & (df["g"] > 100) & df["t_inv"].notna()
    a1.hexbin(df.loc[m, "ta"], df.loc[m, "t_inv"] + np.random.default_rng(0).uniform(-0.5, 0.5, m.sum()),
              gridsize=35, cmap="Blues", mincnt=1, linewidths=0)
    a1.set_xlabel("air temperature, degC")
    a1.set_ylabel("inverter temperature, degC")
    a1.set_title("Fan-regulated: flat against air temperature")
    mo = res.tables["inverter_residual_monthly"]["resid"]
    x = mo.index.to_timestamp() if hasattr(mo.index, "to_timestamp") else pd.to_datetime(mo.index.astype(str))
    a2.axhline(0, color=AXIS, lw=1)
    a2.bar(x, mo.to_numpy(), width=22, color=CELL)
    a2.axvspan(pd.Timestamp("2016-06-04"), pd.Timestamp("2016-07-08"), color=GRID, lw=0)
    a2.text(pd.Timestamp("2016-06-20"), mo.max() * 0.95, "outage", ha="center", fontsize=8, color=INK2)
    a2.set_ylabel("measured - model, K (monthly)")
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))  # quarterly labels nearly touched
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a2.set_title(f"After the July 2016 repair: {s['residual_after_jul2016'] - s['residual_before_jun2016']:+.1f} K")
    d = res.tables["derating_by_air_temperature"]
    xi = np.arange(len(d))
    a3.axhline(0, color=AXIS, lw=1)
    a3.plot(xi, 100 * d["real_flagged_share"], color=CELL, marker="o", ms=6, label="measured output")
    a3.plot(xi, 100 * d["injected_flagged_share"], color=DEFAULT, marker="o", ms=6,
            label="injected: cap 90 kW above 35 degC")
    a3.set_xticks(xi, [i.replace("(", "").replace("]", "").replace(", ", "-") for i in d.index], fontsize=8)
    a3.set_xlabel("air temperature, degC")
    a3.set_ylabel("steps flagged as capped, % (load >= 60 %)")
    a3.set_title(f"Derating search: {s['derating_flagged_days']} flagged days in two years")
    a3.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def fig_transformer(res, path: Path) -> Path:
    from pv_thermal import transformer
    sz = res.tables["transformer_sizing"]
    pac, ta = res.series["hot_week"]["Warsaw"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9))
    tmax = ta.rolling("7D").mean().idxmax()
    sl = slice(tmax - pd.Timedelta(days=5), tmax + pd.Timedelta(days=2))
    for r, c in zip((0.7, 0.8, 1.0), RAMP[::-1]):
        th = transformer.hotspot(pac / (r * 1000 / 1.2), ta)
        a1.plot(th.loc[sl].index, th.loc[sl, "hot_spot"], color=c, lw=1.6, label=f"rating {r:.1f} x inverter")
    a1.plot(ta.loc[sl].index, ta.loc[sl], color=INK2, lw=1.2, label="air")
    a1.axhline(transformer.LIMIT_HOT_SPOT, color=SENSOR, lw=1, ls="--")
    a1.text(ta.loc[sl].index[-1], transformer.LIMIT_HOT_SPOT + 2, "120 degC limit", fontsize=8, color=INK2,
            ha="right")
    a1.set_ylim(top=transformer.LIMIT_HOT_SPOT + 12)
    a1.set_ylabel("hot-spot, degC")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a1.set_title("Warsaw, hottest week of 2022-2024 (DC/AC 1.2)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    for site, g in sz.groupby("site", sort=False):
        if site.startswith("Las Vegas"):
            a2.plot(g["ratio"], g["ageing_days_per_year"], color=SENSOR, marker="o", ms=5, label="Las Vegas")
        elif site == "Warsaw":
            a2.plot(g["ratio"], g["ageing_days_per_year"], color=CELL, marker="o", ms=5, label="Warsaw")
        else:
            a2.plot(g["ratio"], g["ageing_days_per_year"], color=AXIS, lw=1.2, label="other Polish sites"
                    if site == "Gdansk" else None)
    a2.axhline(365, color=INK2, lw=1, ls="--")
    a2.text(0.98, 420, "nominal ageing (365 days/year)", fontsize=8, color=INK2, ha="right")
    a2.set_yscale("log")
    a2.set_xlabel("transformer rating / inverter rating")
    a2.set_ylabel("insulation ageing, days per year")
    a2.set_title("IEC 60076-7 ageing vs transformer size")
    a2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def fig_poland(res, path: Path) -> Path:
    t = res.tables["temperature_loss_by_site"]
    cols = [c for c in t.columns if c not in ("poa_kwh_m2_yr", "mean_air_c")]
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    x = np.arange(len(t))
    w = 0.8 / len(cols)
    for i, (c, colour) in enumerate(zip(cols, CATEGORICAL)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, -100 * t[c], w * 0.92, color=colour, label=c)
    ax.set_xticks(x, [s.replace(" (measured POA)", "\n(11 deg tilt)") for s in t.index])
    ax.set_ylabel("annual temperature loss, %")
    ax.set_title("Annual temperature loss vs 25 degC: Las Vegas and six Polish sites (ERA5 2022-2024, 30 deg tilt)")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def save_all(res, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_site_sensors(res, out_dir / "site_sensors.png"), fig_models(res, out_dir / "module_models.png"),
            fig_voc(res, out_dir / "voc_thermometry.png"), fig_losses_pr(res, out_dir / "losses_pr.png"),
            fig_faults(res, out_dir / "sensor_faults.png"), fig_inverter(res, out_dir / "inverter.png"),
            fig_transformer(res, out_dir / "transformer.png"), fig_poland(res, out_dir / "poland.png")]
