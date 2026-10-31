"""Static HTML dashboard (Plotly.js from the official CDN) for GitHub Pages.

Figures are written as plain Plotly JSON with colour tokens ("@cell", "@sensor", ...) that the page resolves for
the light or dark theme, so one file follows the reader's colour scheme. Colour roles match figures.py.

Usage:
    write_dashboard(res, Path("docs"))
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from pv_thermal import transformer
from pv_thermal.site import SYSTEM_34

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
REPO_URL = "https://github.com/kwwaleszczynski-del/pv-thermal-check"
TZ = SYSTEM_34.tz


def _num(values, nd: int = 3) -> list:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(None if math.isnan(f) else round(f, nd))
    return out


def _t(index) -> list[str]:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(TZ).tz_localize(None)
    return [t.strftime("%Y-%m-%d %H:%M") for t in idx]


def _line(x, y, name, colour, nd=3, mode="lines", **kw) -> dict:
    return {"type": "scatter", "mode": mode, "x": x, "y": _num(y, nd), "name": name,
            "line": {"color": colour, "width": 2}, "marker": {"size": 7, "color": colour}, **kw}


def figures(res) -> dict[str, dict]:
    s, tb, se = res.summary, res.tables, res.series
    figs = {}
    st = pd.Series(s["qc"]["ambient"]["stuck_share_by_month"])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    figs["stuck"] = {"data": [{"type": "bar", "x": [months[int(m) - 1] for m in st.index], "y": _num(100 * st, 1),
                               "marker": {"color": "@sensor"}, "name": "stuck readings",
                               "hovertemplate": "%{x}: %{y:.0f}% of readings<extra></extra>"}],
                     "layout": {"yaxis": {"title": {"text": "% of readings at exactly 0 degF"}}, "bargap": 0.3,
                                "showlegend": False}}
    pr_ratio = pd.Series(s["qc"]["poa_ratio_by_month"])
    figs["poa"] = {"data": [_line([months[int(m) - 1] for m in pr_ratio.index], pr_ratio, "POA / clear-sky",
                                  "@sensor", mode="lines+markers", hovertemplate="%{y:.3f}<extra></extra>")],
                   "layout": {"yaxis": {"title": {"text": "sensor / clear-sky model"}, "range": [0.9, 1.3]},
                              "showlegend": False,
                              "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 1, "y1": 1,
                                          "line": {"color": "@axis", "width": 1}}]}}
    t = tb["module_models"]
    models = list(dict.fromkeys(t["model"]))
    figs["rmse"] = {"data": [
        {"type": "bar", "x": models, "name": label, "marker": {"color": colour},
         "y": _num([t.query("model == @m and variant == @v")["rmse"].iloc[0] for m in models], 2),
         "hovertemplate": "%{x}: %{y:.2f} K<extra>" + label + "</extra>"}
        for v, colour, label in [("default", "@default", "published defaults"),
                                 ("fitted", "@ink2", "fitted to the sensor"),
                                 ("fitted + lag", "@sensor", "fitted + lag")]],
        "layout": {"barmode": "group", "yaxis": {"title": {"text": "RMSE on 2016, K"}}, "bargap": 0.25}}
    b = tb["bias_by_hour"]
    figs["bias"] = {"data": [
        _line(list(b.index), b["static"], "fitted, steady state", "@sensor", 2, "lines+markers",
              hovertemplate="%{y:.2f} K"),
        _line(list(b.index), b["lag"], "fitted + lag", "@ink2", 2, "lines+markers", hovertemplate="%{y:.2f} K")],
        "layout": {"xaxis": {"title": {"text": "local standard time, h"}},
                   "yaxis": {"title": {"text": "model - sensor, K"}}, "hovermode": "x unified"}}
    fr = se["frame"].loc["2016-06-06 12:00":"2016-06-10 06:00"]
    cell = se["voc"]["t_cell_voc"].reindex(fr.index)
    x = _t(fr.index)
    figs["voc_week"] = {"data": [
        _line(x, cell, "cell (open-circuit voltage)", "@cell", 1, "lines+markers", hovertemplate="%{y:.1f} degC",
              marker={"size": 4, "color": "@cell"}),
        _line(x, fr["t_mod"], "back-of-module sensor", "@sensor", 1, hovertemplate="%{y:.1f} degC"),
        _line(x, fr["ta"], "air (airport)", "@ink2", 1, hovertemplate="%{y:.1f} degC")],
        "layout": {"yaxis": {"title": {"text": "degC"}}, "hovermode": "x unified"}}
    v = se["voc"].dropna(subset=["t_cell_voc", "t_mod", "ta"])
    v = v[v["g"] > 20]
    figs["voc_scatter"] = {"data": [
        {"type": "scatter", "mode": "markers", "x": _num(v["g"], 0), "y": _num(v["t_cell_voc"] - v["ta"], 1),
         "name": "cell (Voc)", "marker": {"color": "@cell", "size": 5, "opacity": 0.6},
         "hovertemplate": "%{x:.0f} W/m2: %{y:.1f} K<extra>cell</extra>"},
        {"type": "scatter", "mode": "markers", "x": _num(v["g"], 0), "y": _num(v["t_mod"] - v["ta"], 1),
         "name": "sensor", "marker": {"color": "@sensor", "size": 5, "opacity": 0.6},
         "hovertemplate": "%{x:.0f} W/m2: %{y:.1f} K<extra>sensor</extra>"}],
        "layout": {"xaxis": {"title": {"text": "POA irradiance, W/m2"}},
                   "yaxis": {"title": {"text": "rise over air, K"}}}}
    lo = tb["temperature_loss_monthly"]
    xm = [d.strftime("%Y-%m-01") for d in lo.index.tz_convert(None)]
    figs["loss"] = {"data": [
        _line(xm, 100 * lo["model_cell"], "cell (Voc-calibrated)", "@cell", 2, "lines+markers",
              hovertemplate="%{y:.1f}%"),
        _line(xm, 100 * lo["sensor_plus_3k"], "sensor + 3 K", "@sensor", 2, "lines+markers",
              hovertemplate="%{y:.1f}%")],
        "layout": {"yaxis": {"title": {"text": "loss vs 25 degC, %"}}, "hovermode": "x unified"}}
    pr = tb["pr_monthly"]
    xp = [d.strftime("%Y-%m-01") for d in pr.index.tz_convert(None)]
    figs["pr"] = {"data": [
        _line(xp, pr["pr_raw"], "raw PR", "@ink2", 3, "lines+markers", hovertemplate="%{y:.3f}"),
        _line(xp, pr["pr_stc"], "corrected to 25 degC", "@default", 3, "lines+markers", hovertemplate="%{y:.3f}"),
        _line(xp, pr["pr_nrel"], "NREL weather-corrected", "@cell", 3, "lines+markers", hovertemplate="%{y:.3f}")],
        "layout": {"yaxis": {"title": {"text": "DC performance ratio"}}, "hovermode": "x unified"}}
    fs, fp = se["features_self"], se["features_physics"]
    figs["ratio"] = {"data": [
        _line([d.strftime("%Y-%m-%d") for d in fs.index], fs["ratio"].rolling(7, center=True).median(),
              "vs model fitted to the sensor", "@sensor", 3, hovertemplate="%{y:.2f}"),
        _line([d.strftime("%Y-%m-%d") for d in fp.index], fp["ratio"].rolling(7, center=True).median(),
              "vs physics (Voc-calibrated)", "@cell", 3, hovertemplate="%{y:.2f}")],
        "layout": {"yaxis": {"title": {"text": "daily rise ratio (7-day median)"}, "range": [0, 1.6]},
                   "hovermode": "x unified"}}
    ft = tb["sensor_faults"]
    ft = ft[ft["default_size"]]
    names = {"offset_plus": "+3 K step", "offset_minus": "-3 K step", "drift": "drift 4.5 K / 90 d",
             "detached": "loose sensor (rise x0.6)", "ventilation": "less ventilation (rise x1.3)"}
    yl = [names[f] for f in ft["fault"]]
    figs["delay"] = {"data": [
        {"type": "bar", "orientation": "h", "y": yl, "x": _num(ft["cusum_median_days"], 1), "name": "CUSUM",
         "marker": {"color": "@cell"}, "customdata": _num(100 * ft["cusum_detected"], 0),
         "hovertemplate": "%{x:.0f} days (detected in %{customdata}%)<extra>CUSUM</extra>"},
        {"type": "bar", "orientation": "h", "y": yl, "x": _num(ft["shewhart_median_days"], 1),
         "name": "3-sigma baseline", "marker": {"color": "@axis"}, "customdata": _num(100 * ft["shewhart_detected"], 0),
         "hovertemplate": "%{x:.0f} days (detected in %{customdata}%)<extra>baseline</extra>"}],
        "layout": {"barmode": "group", "xaxis": {"title": {"text": "median days to alarm"}},
                   "yaxis": {"autorange": "reversed"}}}
    mo = tb["inverter_residual_monthly"]["resid"]
    figs["inverter"] = {"data": [{"type": "bar", "x": [f"{p}-01" for p in mo.index.astype(str)],
                                  "y": _num(mo, 2), "marker": {"color": "@cell"}, "name": "monthly mean residual",
                                  "hovertemplate": "%{x|%b %Y}: %{y:+.2f} K<extra></extra>"}],
                        "layout": {"yaxis": {"title": {"text": "measured - model, K"}}, "showlegend": False,
                                   "shapes": [{"type": "rect", "xref": "x", "yref": "paper", "x0": "2016-06-04",
                                               "x1": "2016-07-08", "y0": 0, "y1": 1, "line": {"width": 0},
                                               "fillcolor": "@grid"}]}}
    d = tb["derating_by_air_temperature"]
    xl = [i.replace("(", "").replace("]", "").replace(", ", " to ") for i in d.index]
    figs["derating"] = {"data": [
        _line(xl, 100 * d["real_flagged_share"], "measured output", "@cell", 2, "lines+markers",
              hovertemplate="%{y:.1f}%"),
        _line(xl, 100 * d["injected_flagged_share"], "injected cap (90 kW above 35 degC)", "@default", 2,
              "lines+markers", hovertemplate="%{y:.1f}%")],
        "layout": {"xaxis": {"title": {"text": "air temperature, degC"}, "type": "category"},
                   "yaxis": {"title": {"text": "steps flagged as capped, %"}}, "hovermode": "x unified"}}
    pac, ta = se["hot_week"]["Warsaw"]
    tmax = ta.rolling("7D").mean().idxmax()
    sl = slice(tmax - pd.Timedelta(days=5), tmax + pd.Timedelta(days=2))
    xw = [t.strftime("%Y-%m-%d %H:%M") for t in ta.loc[sl].index.tz_convert("Europe/Warsaw").tz_localize(None)]
    data = []
    for r, c in zip((0.7, 0.8, 1.0), ("@r2", "@r1", "@r0")):
        th = transformer.hotspot(pac / (r * 1000 / 1.2), ta)
        data.append(_line(xw, th.loc[sl, "hot_spot"], f"rating {r:.1f} x inverter", c, 1,
                          hovertemplate="%{y:.0f} degC"))
    data.append(_line(xw, ta.loc[sl], "air", "@ink2", 1, hovertemplate="%{y:.1f} degC"))
    figs["tr_week"] = {"data": data, "layout": {
        "yaxis": {"title": {"text": "hot-spot, degC"}}, "hovermode": "x unified",
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 120, "y1": 120,
                    "line": {"color": "@sensor", "width": 1, "dash": "dash"}}]}}
    sz = tb["transformer_sizing"]
    data = []
    for site, g in sz.groupby("site", sort=False):
        colour = "@sensor" if site.startswith("Las Vegas") else "@cell" if site == "Warsaw" else "@axis"
        data.append(_line(list(g["ratio"]), g["ageing_days_per_year"], site.replace(" (measured POA)", ""), colour,
                          1, "lines+markers", hovertemplate="%{y:.3g} days/yr",
                          marker={"size": 6 if colour != "@axis" else 4, "color": colour}))
    figs["tr_ageing"] = {"data": data, "layout": {
        "xaxis": {"title": {"text": "transformer rating / inverter rating"}},
        "yaxis": {"title": {"text": "ageing, days per year"}, "type": "log"}, "hovermode": "x unified",
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 365, "y1": 365,
                    "line": {"color": "@ink2", "width": 1, "dash": "dash"}}]}}
    lt = tb["temperature_loss_by_site"]
    cols = [c for c in lt.columns if c not in ("poa_kwh_m2_yr", "mean_air_c")]
    sites = [i.replace(" (measured POA)", "") for i in lt.index]
    figs["poland"] = {"data": [
        {"type": "bar", "x": sites, "y": _num(-100 * lt[c], 2), "name": c, "marker": {"color": f"@c{i + 1}"},
         "hovertemplate": "%{x}: %{y:.2f}%<extra>" + c + "</extra>"} for i, c in enumerate(cols)],
        "layout": {"barmode": "group", "yaxis": {"title": {"text": "annual temperature loss, %"}}, "bargap": 0.2}}
    return figs


def _tile(label: str, value: str, sub: str) -> str:
    return (f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}'
            f'</div><div class="sub">{html.escape(sub)}</div></div>')


def kpi_tiles(res) -> str:
    s = res.summary
    m, v, p, f, i, pl = s["models"], s["voc"], s["performance"], s["faults"], s["inverter"], s["poland"]
    ft = res.tables["sensor_faults"]
    ft = ft[ft["default_size"]].set_index("fault")
    sm = pl["transformer_smallest_ratio"]
    pl_ratios = [r for k, r in sm.items() if not k.startswith("Las Vegas")]
    tiles = [
        _tile("Sensor sees of the cell's heating", f"{100 * v['sensor_gain']:.0f} %",
              f"cells {v['cell_minus_sensor_g800']:.0f} K hotter than the sensor above 800 W/m2 (Voc thermometry)"),
        _tile("Module model error (best, test year)", f"{m['best_rmse']:.1f} K",
              f"published defaults: {min(m['rmse_default'].values()):.1f} to {max(m['rmse_default'].values()):.1f} K"),
        _tile("Temperature loss, Las Vegas", f"{100 * _year(p['annual_loss_model'], 2017):.1f} %",
              f"the sensor would say {100 * _year(p['annual_loss_sensor'], 2017):.1f} %"),
        _tile("Seasonal PR spread", f"{100 * p['seasonality']['pr_raw']['spread']:.1f} -> "
              f"{100 * p['seasonality']['pr_nrel']['spread']:.1f} pp", "raw vs NREL weather-corrected, monthly"),
        _tile("Sensor faults caught", f"{ft.loc['offset_plus', 'cusum_median_days']:.0f}-"
              f"{ft.loc['drift', 'cusum_median_days']:.0f} days",
              f"median delay, all injected faults found; {len(f['clean_alarms_2016'])} alarm"
              f"{'' if len(f['clean_alarms_2016']) == 1 else 's'} in the clean test year"),
        _tile("Inverter", f"{i['t_inv_operating_range'][0]:.0f}-{i['t_inv_operating_range'][1]:.0f} degC",
              f"fan-regulated; {i['derating_flagged_days']} derating days up to {i['max_air_temp_operating']:.0f} degC air"),
        _tile("Temperature loss, Poland", f"{-100 * pl['loss_best_range'][1]:.1f}-{-100 * pl['loss_best_range'][0]:.1f} %",
              f"6 sites, ERA5 2022-2024; models differ by {pl['model_spread_pp']:.1f} pp on average"),
        _tile("Smallest PV transformer", f"{min(pl_ratios):.2f}-{max(pl_ratios):.2f} x",
              f"of inverter rating in Poland ({sm['Las Vegas (measured POA)']:.2f} x in Las Vegas), IEC 60076-7"),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _year(d: dict, y: int):
    return d[y] if y in d else d[str(y)]


def poland_table(res) -> str:
    lt = res.tables["temperature_loss_by_site"]
    sz = res.summary["poland"]["transformer_smallest_ratio"]
    cols = [c for c in lt.columns if c not in ("poa_kwh_m2_yr", "mean_air_c")]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["POA kWh/m2/yr", "mean air degC", *cols,
                                                         "smallest transformer"])
    rows = []
    for site, r in lt.iterrows():
        cells = [f"{r['poa_kwh_m2_yr']:.0f}", f"{r['mean_air_c']:.1f}"] + [f"{-100 * r[c]:.2f} %" for c in cols]
        cells.append(f"{sz[site]:.2f} x")
        rows.append(f"<tr><th>{html.escape(site)}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<table><thead><tr><th>Site</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def write_dashboard(res, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = res.summary
    v, i = s["voc"], s["inverter"]
    page = TEMPLATE
    rep = {
        "%%PLOTLY%%": PLOTLY_JS, "%%REPO%%": REPO_URL, "%%KPIS%%": kpi_tiles(res),
        "%%POA%%": f"{100 * (s['qc']['poa_scale'] - 1):.0f}",
        "%%VOCDAYS%%": str(v["open_circuit_days"]),
        "%%UC%%": f"{v['pvsyst_oc']['u_c']:.1f}", "%%UV%%": f"{v['pvsyst_oc']['u_v']:.2f}",
        "%%INVSTEP%%": f"{i['residual_after_jul2016'] - i['residual_before_jun2016']:+.1f}",
        "%%TABLE%%": poland_table(res),
        "%%FIGURES%%": json.dumps(figures(res), separators=(",", ":")).replace("</", "<\\/"),
    }
    for k, val in rep.items():
        page = page.replace(k, val)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PV Thermal Check</title>
<meta name="description" content="Module, inverter and transformer temperatures of a PV plant on open data.">
<script src="%%PLOTLY%%" charset="utf-8"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0 0 4px; }
p { margin: 4px 0 12px; color: var(--ink2); }
a { color: var(--accent); }
.lead { max-width: 860px; }
.meta { font-size: 13px; color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { font-size: 13px; color: var(--ink2); }
.tile .value { font-size: 28px; font-weight: 600; margin: 2px 0; }
.tile .sub { font-size: 12px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-top: 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 6px;
  min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.chart { width: 100%; height: 340px; }
h2.section { margin: 32px 0 0; font-size: 20px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
thead th { color: var(--ink2); font-weight: 600; }
tbody th { text-align: left; font-weight: 600; }
footer { margin-top: 32px; font-size: 13px; color: var(--muted); }
@media (max-width: 520px) { .grid { grid-template-columns: 1fr; } .chart { height: 300px; } }
</style>
</head>
<body>
<main>
<h1>PV thermal check</h1>
<p class="lead">Module, inverter and transformer temperatures of a real PV plant (NREL PVDAQ system 34, a 147 kWp
rooftop array in Las Vegas, 2016-2017): which thermal model fits, whether the temperature sensors can be trusted,
what heat costs in energy and performance ratio, how to catch a failing sensor, and what the fitted physics
implies in Polish weather. <a href="%%REPO%%">Code and method</a></p>
%%KPIS%%
<h2 class="section">1. What the site sensors say</h2>
<div class="grid">
<section class="card"><h2>Ambient sensor stuck at 0 degF</h2><p>Share of readings exactly at 0 degF by month; the
analysis uses the airport station (NOAA ISD, 14 km) for air temperature and 10 m wind instead.</p>
<div id="stuck" class="chart"></div></section>
<section class="card"><h2>POA sensor reads high</h2><p>Measured / clear-sky POA on clear days: the sensor reads
about %%POA%% % above the model, so irradiance is rescaled before any fitting.</p><div id="poa" class="chart"></div>
</section>
</div>
<h2 class="section">2. Module temperature models</h2>
<div class="grid">
<section class="card"><h2>Test error (fitted on 2017, scored on 2016)</h2><p>Against the back-of-module sensor.
Published defaults overpredict this sensor by several kelvin.</p><div id="rmse" class="chart"></div></section>
<section class="card"><h2>Error by time of day</h2><p>Mornings are too warm in every steady-state model: after a
clear desert night the module starts below air temperature. A first-order lag helps a little.</p>
<div id="bias" class="chart"></div></section>
</div>
<h2 class="section">3. Is the module sensor right? Voc thermometry</h2>
<div class="grid">
<section class="card"><h2>Inverter down, array at open circuit</h2><p>During the June 2016 outage the DC voltage
channel reads the string open-circuit voltage, which gives the cell temperature through the module's voltage
coefficient (%%VOCDAYS%% days with open-circuit data in total).</p><div id="voc_week" class="chart"></div></section>
<section class="card"><h2>Rise over air vs irradiance</h2><p>The sensor reports a fraction of the heating the
cells really see. Fitted to the cells: PVsyst U<sub>c</sub> = %%UC%%, U<sub>v</sub> = %%UV%% W/m2K.</p>
<div id="voc_scatter" class="chart"></div></section>
</div>
<h2 class="section">4. Temperature losses and performance ratio</h2>
<div class="grid">
<section class="card"><h2>Monthly temperature loss</h2><p>Irradiance-weighted loss versus 25 degC with the module's
-0.458 %/K power coefficient.</p><div id="loss" class="chart"></div></section>
<section class="card"><h2>Monthly DC performance ratio</h2><p>Raw PR falls every summer. Correcting to 25 degC
(IEC 61724-1) or to the plant's mean cell temperature (NREL) takes most of the seasonal swing out. Sun above
30 degrees only, to keep winter row shading out.</p><div id="pr" class="chart"></div></section>
</div>
<h2 class="section">5. Sensor fault detection</h2>
<div class="grid">
<section class="card"><h2>The real sensor</h2><p>Daily ratio of measured to modelled temperature rise. Against a
model fitted to its own history the sensor looks healthy; against the Voc-calibrated physics it has been reading
low all along.</p><div id="ratio" class="chart"></div></section>
<section class="card"><h2>Injected faults: time to alarm</h2><p>Faults injected into the real 2016 record at 40
start dates; CUSUM on the daily residual and rise ratio, threshold set for at most one false alarm per year.</p>
<div id="delay" class="chart"></div></section>
</div>
<h2 class="section">6. Inverter</h2>
<div class="grid">
<section class="card"><h2>Inverter temperature residual</h2><p>Model: temperature = constant + load term + air term;
the air term is zero (fan-regulated cooling). After the repair that ended the June 2016 outage the unit runs
%%INVSTEP%% K cooler at the same load.</p><div id="inverter" class="chart"></div></section>
<section class="card"><h2>Thermal derating search</h2><p>A step is flagged when AC power sits on a plateau more
than 8 % below the expected output. The real record shows no capped days; an injected cap is found.</p>
<div id="derating" class="chart"></div></section>
</div>
<h2 class="section">7. Step-up transformer (IEC 60076-7 model)</h2>
<div class="grid">
<section class="card"><h2>Hot-spot in Warsaw's hottest week</h2><p>Generic ONAN distribution transformer fed by a
PV plant with DC/AC 1.2 (ERA5 weather), for three transformer ratings.</p><div id="tr_week" class="chart"></div>
</section>
<section class="card"><h2>Insulation ageing vs transformer size</h2><p>Days of life consumed per year (365 = nominal
rate). PV loads a transformer only around midday, so it can be smaller than the inverter.</p>
<div id="tr_ageing" class="chart"></div></section>
</div>
<h2 class="section">8. Transfer to Poland</h2>
<section class="card wide" style="margin-top:16px"><h2>Annual temperature loss by site and model</h2><p>Six Polish
sites, ERA5 2022-2024, 30 degree tilt; Las Vegas (measured, 11 degree tilt) for reference. The fitted coefficients
belong to one rooftop array: another mounting gives other coefficients, so treat the Polish numbers as an
illustration.</p><div id="poland" class="chart" style="height:380px"></div>
<div class="table-wrap">%%TABLE%%</div></section>
<footer>
<p>Data: NREL PVDAQ (Deline et al. 2021, doi:10.25984/1846021, CC BY 4.0); NOAA ISD-Lite (public domain); ERA5
(Copernicus Climate Change Service) via the Open-Meteo archive API (CC BY 4.0). Generic transformer parameters from
the IEC 60076-7 loading guide; real units differ. Not engineering or investment advice.</p>
</footer>
</main>
<script>
const FIGS = %%FIGURES%%;
const THEMES = {
  light: { surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
    cell: "#2a78d6", sensor: "#eb6834", default: "#1baf7a",
    c1: "#2a78d6", c2: "#eb6834", c3: "#1baf7a", c4: "#eda100", c5: "#e87ba4", c6: "#008300",
    r0: "#86b6ef", r1: "#2a78d6", r2: "#104281" },
  dark: { surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
    cell: "#3987e5", sensor: "#d95926", default: "#199e70",
    c1: "#3987e5", c2: "#d95926", c3: "#199e70", c4: "#c98500", c5: "#d55181", c6: "#008300",
    r0: "#184f95", r1: "#3987e5", r2: "#9ec5f4" },
};
function themeName() {
  const forced = document.documentElement.dataset.theme;
  if (forced === "light" || forced === "dark") return forced;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function resolve(value, t) {
  if (typeof value === "string" && value.startsWith("@") && value.slice(1) in t) return t[value.slice(1)];
  if (Array.isArray(value)) return value.map(v => resolve(v, t));
  if (value && typeof value === "object") {
    const out = {};
    for (const k in value) out[k] = resolve(value[k], t);
    return out;
  }
  return value;
}
function axis(t, extra) {
  return Object.assign({ gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted },
    title: { font: { color: t.ink2, size: 12 } }, automargin: true }, extra || {});
}
function render() {
  if (!window.Plotly) return;
  const t = THEMES[themeName()];
  for (const id in FIGS) {
    const el = document.getElementById(id);
    if (!el) continue;
    const fig = resolve(FIGS[id], t);
    const lay = fig.layout || {};
    const layout = Object.assign({}, lay, {
      paper_bgcolor: t.surface, plot_bgcolor: t.surface,
      font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.ink, size: 12 },
      margin: { l: 56, r: 16, t: 8, b: 40 },
      legend: { orientation: "h", y: -0.22, font: { color: t.ink2 } },
      hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
      xaxis: axis(t, lay.xaxis), yaxis: axis(t, lay.yaxis),
    });
    if (window.innerWidth < 640) {
      // phones: slanted, smaller x labels (categories collide otherwise) and the legend below the axis title
      layout.xaxis = Object.assign({}, layout.xaxis, { tickangle: -45, tickfont: Object.assign({}, layout.xaxis.tickfont, { size: 9.5 }) });
      layout.yaxis = Object.assign({}, layout.yaxis, { tickfont: Object.assign({}, layout.yaxis.tickfont, { size: 9.5 }) });
      layout.legend = Object.assign({}, layout.legend, { y: -0.45, yanchor: "top" });
      layout.margin = Object.assign({}, layout.margin, { b: 56 });
    }
    Plotly.react(el, fig.data, layout, { responsive: true, displaylogo: false });
  }
}
render();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""
