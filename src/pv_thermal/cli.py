"""Command line entry point.

Usage:
    python -m pv_thermal run                       # download (cached), analyse, write results/, figures, dashboard
    python -m pv_thermal run --pvdaq-dir ../pv-anomaly-autoencoder/data/pvdaq   # reuse PVDAQ files on disk
    python -m pv_thermal run --no-figures --no-dashboard
    python -m pv_thermal fetch                     # downloads only
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

log = logging.getLogger("pv_thermal")


def _fetch(args) -> None:
    from pv_thermal import data, poland
    from pv_thermal.site import ISD_STATION, POLAND, POLAND_YEARS, PVDAQ_SYSTEM, YEARS
    pvdaq_dir = args.pvdaq_dir or args.data_dir / "pvdaq"
    data.fetch_pvdaq(PVDAQ_SYSTEM, f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", pvdaq_dir)
    data.load_isd(*ISD_STATION, list(YEARS), args.data_dir / "isd")
    poland.fetch_era5(POLAND, f"{POLAND_YEARS[0]}-01-01", f"{POLAND_YEARS[1]}-12-31", args.data_dir / "era5")


def _run(args) -> None:
    from pv_thermal import analysis
    res = analysis.run_all(args.data_dir, args.results_dir, args.pvdaq_dir)
    if not args.no_figures:
        from pv_thermal import figures
        for p in figures.save_all(res, args.docs_dir / "img"):
            log.info("figure %s", p)
    if not args.no_dashboard:
        from pv_thermal import dashboard
        log.info("dashboard %s", dashboard.write_dashboard(res, args.docs_dir))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-thermal", description="Module, inverter and transformer temperatures of "
                                "a PV plant on open data: model fits, sensor checks, losses, PR and a Polish transfer.")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="download cache (default: data)")
    p.add_argument("--pvdaq-dir", type=Path, default=None,
                   help="folder holding system_34/ daily PVDAQ files (default: <data-dir>/pvdaq)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download all inputs into the cache")
    r = sub.add_parser("run", help="run the full analysis")
    r.add_argument("--results-dir", type=Path, default=Path("results"))
    r.add_argument("--docs-dir", type=Path, default=Path("docs"))
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--no-dashboard", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    {"fetch": _fetch, "run": _run}[args.command](args)


if __name__ == "__main__":
    main()
