"""
compare_flowsheets.py
---------------------
Runs Membrane, PSA, and AmineWash flowsheets at a common syngas target and
compares them side-by-side on cost, energy, and process metrics.

Results are printed as formatted tables and saved as bar charts in the
../Plots/ directory.

Configuration
-------------
Edit the constants at the top of the file to change the syngas target,
H2 sourcing type, or homotopy steps.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Reassigning the non-component attribute")
warnings.filterwarnings("ignore", category=UserWarning, module="pyomo")

import logging

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyomo.environ as pyo
from pyomo.environ import value
from prettytable import PrettyTable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # FullFlowsheets/Scripts/
_PYTHON_ROOT = _HERE.parent.parent.parent  # Python/
_PLOTS_DIR = _HERE.parent / "Plots" / "flowsheet_comparison"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_PYTHON_ROOT))
sys.path.insert(0, str(_PYTHON_ROOT / "SubModels"))

from SubModels.MembraneFlowsheet import MembraneFlowsheet  # noqa: E402
from SubModels.PSAFlowsheet import PSAFlowsheet  # noqa: E402
from SubModels.AmineWashFlowsheet import AmineWashFlowsheet  # noqa: E402
from configuration import Economics  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
SYNGAS_TARGET_TPD = 75.0  # t/day
SYNGAS_RATIO = 2.0  # H2/CO molar ratio
H2_TYPE = "H2Electrolyzer"  # "H2Electrolyzer" | "Buy" | "Electrolyzer"
PURGE_FRACTION = 0.02  # 2% purge → 98% recycle

# Known-good electrolyzer operating point
_E_APPL = -1.31
_V = 0.08
_CO2SHARE = 0.90

HOMOTOPY_STEPS = [40.0, 55.0, 65.0, SYNGAS_TARGET_TPD]  # t/day

# ---------------------------------------------------------------------------
# Pseudo-global solve via multistart
# ---------------------------------------------------------------------------
# MAiNGO / true global solvers aren't available here, so we approximate by
# running repeated IPOPT restarts (pyomo.contrib.multistart) around the
# converged incumbent and keeping the best objective. Applied only on the
# FINAL homotopy step — the earlier steps still warm-start to build a good
# incumbent for the restarts to perturb from.
USE_MULTISTART = True
MULTISTART_KWARGS = {
    "strategy": "rand_guess_and_bound",  # perturb from incumbent toward bounds
    "iterations": 20,  # number of restarts ("spam")
}

# Set by sensitivity_sweep.py to annotate individual plot titles (e.g. "price_power = 0.04 €/kWh")
SWEEP_SUBTITLE: str = ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subtitle() -> str:
    """Return a parenthesised sweep annotation if one is active, else empty string."""
    return f"  [{SWEEP_SUBTITLE}]" if SWEEP_SUBTITLE else ""


def _kgs(tpd: float) -> float:
    return tpd * 1000.0 / 86400.0


def _homotopy_solve(
    m, targets, show_log: bool = False, multistart: bool = False, multistart_kwargs: dict | None = None
):
    results = None
    n = len(targets)
    for i, tgt in enumerate(targets):
        m.mf_syngas_target.set_value(_kgs(tgt))  # type: ignore[operator]
        use_ms = multistart and (i == n - 1)  # only on the final step
        if i == 0:
            results = m.solve(  # type: ignore[operator]
                show_log=show_log,
                multistart=use_ms,
                multistart_kwargs=multistart_kwargs,
            )
        else:
            results = m.solve(  # type: ignore[operator]
                show_log=show_log,
                solver_options={
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": 1e-9,
                    "warm_start_mult_bound_push": 1e-9,
                    "mu_init": 1e-3,
                },
                multistart=use_ms,
                multistart_kwargs=multistart_kwargs,
            )
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  [homotopy {tgt:.0f} t/day] -> {tc}")
    return results


def _solver_ok(results) -> bool:
    if results is None:
        return False
    tc = results.solver.termination_condition
    return tc in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    )


# ---------------------------------------------------------------------------
# Build & solve each flowsheet
# ---------------------------------------------------------------------------


def build_membrane() -> MembraneFlowsheet:
    print("\n=== Building MembraneFlowsheet ===")
    m = MembraneFlowsheet(
        n_stages=2,
        with_flash=False,
        membrane_material="PEBAX",
        purge_fraction=PURGE_FRACTION,
        float_purge=False,
        recycle_permeate_H2=False,
    )
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=SYNGAS_TARGET_TPD,
        syngas_ratio=SYNGAS_RATIO,
        H2_type=H2_TYPE,
        price_CO2_feed=Economics().price_CO2_feed,
    )
    m.initialize_electrolyzer(E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE)  # type: ignore[operator]
    # Force a genuine intercooled 2-stage compression train: equal pressure
    # ratios across the two stages (P_inter^2 = Pin * Pout) plus full intercooling
    # (intercooler outlet fixed to 303.15 K). Left free, the optimizer collapses
    # stage 2 to idle because compression is economically minor — but a single
    # stage to ~18-22 bar implies a ~365 °C discharge, beyond real compressor
    # limits, so a 2-stage train is the physically realistic configuration.
    m.comp.eq_balanced_pr = pyo.Constraint(  # type: ignore[attr-defined]
        expr=m.comp._P_inter[0] ** 2 == m.comp._comps[0].Pin * m.comp.Pout_var
    )
    m.comp._T_ic_out[0].fix(303.15)
    results = _homotopy_solve(m, HOMOTOPY_STEPS, multistart=USE_MULTISTART, multistart_kwargs=MULTISTART_KWARGS)
    if _solver_ok(results):
        print("  -> Membrane flowsheet: SOLVED")
    else:
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  -> Membrane flowsheet: WARNING — {tc}")
    return m


def build_psa() -> PSAFlowsheet:
    """PSA standalone test uses purge_fraction=0 and a direct single solve —
    homotopy warm-starts push IPOPT into infeasible basins for this flowsheet."""
    print("\n=== Building PSAFlowsheet ===")
    m = PSAFlowsheet(purge_fraction=0.1, float_purge=False)
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=SYNGAS_TARGET_TPD,
        syngas_ratio=SYNGAS_RATIO,
        H2_type=H2_TYPE,
        price_CO2_feed=Economics().price_CO2_feed,
    )
    m.initialize()  # type: ignore[operator]
    results = m.solve(  # type: ignore[operator]
        show_log=False,
        multistart=USE_MULTISTART,
        multistart_kwargs=MULTISTART_KWARGS,
    )
    if _solver_ok(results):
        print("  -> PSA flowsheet: SOLVED")
    else:
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  -> PSA flowsheet: WARNING — {tc}")
    return m


def build_amine() -> AmineWashFlowsheet:
    print("\n=== Building AmineWashFlowsheet ===")
    m = AmineWashFlowsheet(purge_fraction=PURGE_FRACTION, float_purge=False)
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=SYNGAS_TARGET_TPD,
        syngas_ratio=SYNGAS_RATIO,
        H2_type=H2_TYPE,
        price_CO2_feed=Economics().price_CO2_feed,
    )
    m.initialize_electrolyzer(E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE)  # type: ignore[operator]
    results = _homotopy_solve(m, HOMOTOPY_STEPS, multistart=USE_MULTISTART, multistart_kwargs=MULTISTART_KWARGS)
    if _solver_ok(results):
        print("  -> Amine flowsheet: SOLVED")
    else:
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  -> Amine flowsheet: WARNING — {tc}")
    return m


# ---------------------------------------------------------------------------
# Extract comparable metrics
# ---------------------------------------------------------------------------


def _safe(expr):
    """Return float value or NaN on failure."""
    try:
        return float(value(expr))
    except Exception:
        return float("nan")


def extract_metrics(mem: MembraneFlowsheet, psa: PSAFlowsheet, amine: AmineWashFlowsheet) -> dict:
    """Return a dict-of-dicts: metric_name -> {Membrane, PSA, Amine}."""

    def _syngas_tpd(m):
        return _safe(m.mf_syngas) * 86400.0 / 1000.0

    def _co2_fresh_tpd(m):
        # mol/s * 44 g/mol -> kg/s -> t/day
        return _safe(m.n_CO2_fresh) * 44e-3 * 86400.0 / 1000.0

    def _elec_power_mw(m):
        try:
            return _safe(m.elec.Power) / 1e6
        except Exception:
            return float("nan")

    def _comp_power_mw(m):
        try:
            # MembraneFlowsheet / PSAFlowsheet store it differently
            total = 0.0
            if hasattr(m, "comp"):
                for c in m.comp._comps:
                    total += _safe(c.power_kw) * 1e-3
            elif hasattr(m, "psa"):
                total += _safe(m.psa.power_comp_in) * 1e-3
                total += _safe(m.psa.power_pump_in) * 1e-3
            return total
        except Exception:
            return float("nan")

    def _h2elec_power_mw(m):
        if hasattr(m, "h2_elec"):
            return _safe(m.h2_elec.power_hp_mw)
        return 0.0

    def _amine_heat_gj_tco2(a):
        try:
            return _safe(a.amine.Q_reboiler_spec)
        except Exception:
            return float("nan")

    def _amine_elec_kwh_tco2(a):
        try:
            return _safe(a.amine.W_elec_spec)
        except Exception:
            return float("nan")

    metrics = {
        # ---- Process ----
        "Syngas output [t/day]": {k: _syngas_tpd(v) for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]},
        "Fresh CO₂ feed [t/day]": {
            k: _co2_fresh_tpd(v) for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
        "CO₂ feed / syngas ratio [-]": {},
        # ---- Costs ----
        "CAPEX [M€]": {k: _safe(v.total_capex) / 1e6 for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]},
        "OPEX [M€/yr]": {
            k: _safe(v.total_opex) * 8760 / 1e6 for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
        "Lifecycle cost [M€]": {
            k: _safe(v.total_lifecycle_cost) / 1e6 for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
        # ---- Energy ----
        "CO₂ electrolyzer power [MW]": {
            k: _elec_power_mw(v) for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
        "H₂ electrolyzer power [MW]": {
            k: _h2elec_power_mw(v) for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
        "Compression/sep. power [MW]": {
            k: _comp_power_mw(v) for k, v in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
        },
    }

    # Derived ratio
    for fs, m in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]:
        sg = metrics["Syngas output [t/day]"][fs]
        co2 = metrics["Fresh CO₂ feed [t/day]"][fs]
        metrics["CO₂ feed / syngas ratio [-]"][fs] = co2 / sg if sg else float("nan")

    # Amine-specific heat penalty (only makes sense for amine)
    metrics["Amine reboiler heat [GJ/tCO₂]"] = {
        "Membrane": float("nan"),
        "PSA": float("nan"),
        "Amine": _amine_heat_gj_tco2(amine),
    }
    metrics["Amine elec. penalty [kWh/tCO₂]"] = {
        "Membrane": float("nan"),
        "PSA": float("nan"),
        "Amine": _amine_elec_kwh_tco2(amine),
    }

    return metrics


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------


def print_comparison(metrics: dict):
    print("\n" + "=" * 70)
    print(f"  FLOWSHEET COMPARISON  —  target {SYNGAS_TARGET_TPD:.0f} t/day syngas,  H2: {H2_TYPE}")
    print("=" * 70)

    t = PrettyTable()
    t.field_names = ["Metric", "Membrane", "PSA", "Amine"]
    t.align = "r"
    t.align["Metric"] = "l"

    for name, vals in metrics.items():
        row = [name]
        for fs in ("Membrane", "PSA", "Amine"):
            v = vals.get(fs, float("nan"))
            if v != v:  # NaN
                row.append("—")
            else:
                row.append(f"{v:.3g}")
        t.add_row(row)

    print(t)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = {"Membrane": "#4C72B0", "PSA": "#DD8452", "Amine": "#55A868"}
FLOWSHEETS = ["Membrane", "PSA", "Amine"]


def _bar_chart(ax, metric_vals: dict, ylabel: str, title: str):
    vals = [metric_vals.get(fs, float("nan")) for fs in FLOWSHEETS]
    bars = ax.bar(
        FLOWSHEETS, vals, color=[COLORS[fs] for fs in FLOWSHEETS], edgecolor="white", linewidth=0.8, width=0.3
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    for bar, v in zip(bars, vals):
        if v == v:  # not NaN
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cost_comparison(metrics: dict):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(
        f"Cost Comparison — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=12,
        fontweight="bold",
    )
    _bar_chart(axes[0], metrics["CAPEX [M€]"], "M€", "CAPEX")
    _bar_chart(axes[1], metrics["OPEX [M€/yr]"], "M€/yr", "Annual OPEX")
    _bar_chart(axes[2], metrics["Lifecycle cost [M€]"], "M€", "Lifecycle Cost")
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_costs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


def plot_energy_comparison(metrics: dict):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(
        f"Energy Comparison — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=12,
        fontweight="bold",
    )
    _bar_chart(axes[0], metrics["CO₂ electrolyzer power [MW]"], "MW", "CO₂ Electrolyzer")
    _bar_chart(axes[1], metrics["H₂ electrolyzer power [MW]"], "MW", "H₂ Electrolyzer")
    _bar_chart(axes[2], metrics["Compression/sep. power [MW]"], "MW", "Compression / Separation")
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_energy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_process_comparison(metrics: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"Process Comparison — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=12,
        fontweight="bold",
    )
    _bar_chart(axes[0], metrics["Fresh CO₂ feed [t/day]"], "t/day", "Fresh CO₂ Feed")
    _bar_chart(axes[1], metrics["CO₂ feed / syngas ratio [-]"], "t CO₂ / t syngas", "CO₂ Feed Intensity")
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_process.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_cost_breakdown_stacked(mem, psa, amine):
    """Stacked bar: CAPEX by equipment group for each flowsheet."""

    def _capex_breakdown(m):
        sep = 0.0
        try:
            if hasattr(m, "comp"):
                sep += _safe(m.comp.total_capex)
            if hasattr(m, "mem"):
                sep += _safe(m.mem.CAPEX)
            if hasattr(m, "psa"):
                sep += _safe(m.psa.CAPEX)
            if hasattr(m, "amine"):
                sep += _safe(m.amine.CAPEX)
            if hasattr(m, "flash_CAPEX"):
                sep += _safe(m.flash_CAPEX)
        except Exception:
            pass
        return {
            "CO₂ electrolyzer": _safe(m.elec.CAPEX),
            "H₂ electrolyzer": _safe(m.h2_elec.CAPEX) if hasattr(m, "h2_elec") else 0.0,
            "Separation": sep,
        }

    breakdowns = {
        "Membrane": _capex_breakdown(mem),
        "PSA": _capex_breakdown(psa),
        "Amine": _capex_breakdown(amine),
    }

    _CAT_COLORS_CAPEX = {
        "CO₂ electrolyzer": "#2166ac",
        "H₂ electrolyzer": "#d6604d",
        "Separation": "#4dac26",
    }

    # Fixed stacking order: bottom → top
    all_cats = ["CO₂ electrolyzer", "H₂ electrolyzer", "Separation"]

    _BAR_W = 0.3
    _xs = [i * _BAR_W * 2 for i in range(len(FLOWSHEETS))]
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0.0] * len(FLOWSHEETS)

    for cat in all_cats:
        heights = [breakdowns[fs].get(cat, 0.0) / 1e6 for fs in FLOWSHEETS]
        ax.bar(
            _xs,
            heights,
            bottom=bottoms,
            label=cat,
            color=_CAT_COLORS_CAPEX[cat],
            edgecolor="white",
            linewidth=0.6,
            width=_BAR_W,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=10)
    ax.set_ylabel("CAPEX [M€]", fontsize=10)
    ax.set_title(
        f"CAPEX Breakdown — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_capex_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_opex_breakdown_stacked(mem, psa, amine):
    """Stacked bar: annual OPEX by cost category for each flowsheet."""
    op_hours = Economics().operating_hours_per_year

    def _opex_breakdown(m):
        _econ = Economics()
        # CO2 electrolyzer: recompute electricity and maintenance separately
        # (elec.OPEX [€/h] combines both, so derive from primary variables)
        co2_elec_elec = _safe(m.elec.Power) * _econ.price_power / 1000 * op_hours
        co2_elec_maint = _safe(m.elec.maint_factor) * _safe(m.elec.CAPEX)
        # H2 electrolyzer: OPEX=electricity [€/yr], MAINTENANCE [€/yr] already split
        h2_elec_elec = _safe(m.h2_elec.OPEX) if hasattr(m, "h2_elec") else 0.0
        h2_elec_maint = _safe(m.h2_elec.MAINTENANCE) if hasattr(m, "h2_elec") else 0.0
        feedstock = _safe(m.cost_CO2_feed) if hasattr(m, "cost_CO2_feed") else 0.0
        sep = _safe(m.total_opex) - co2_elec_elec - co2_elec_maint - h2_elec_elec - h2_elec_maint - feedstock
        return {
            "CO₂ elec. electricity": co2_elec_elec,
            "CO₂ elec. maintenance": co2_elec_maint,
            "H₂ elec. electricity": h2_elec_elec,
            "H₂ elec. maintenance": h2_elec_maint,
            "Separation": sep,
            "Feedstock (CO₂)": feedstock,
        }

    breakdowns = {
        "Membrane": _opex_breakdown(mem),
        "PSA": _opex_breakdown(psa),
        "Amine": _opex_breakdown(amine),
    }

    # Fixed stacking order: bottom → top
    all_cats = [
        "CO₂ elec. electricity",
        "CO₂ elec. maintenance",
        "H₂ elec. electricity",
        "H₂ elec. maintenance",
        "Separation",
        "Feedstock (CO₂)",
    ]

    _CAT_COLORS = {
        "CO₂ elec. electricity": "#2166ac",
        "CO₂ elec. maintenance": "#92c5de",
        "H₂ elec. electricity": "#d6604d",
        "H₂ elec. maintenance": "#f4a582",
        "Separation": "#4dac26",
        "Feedstock (CO₂)": "#b8b8b8",
    }

    _BAR_W = 0.3
    _xs = [i * _BAR_W * 2 for i in range(len(FLOWSHEETS))]
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0.0] * len(FLOWSHEETS)

    for cat in all_cats:
        heights = [breakdowns[fs].get(cat, 0.0) / 1e6 for fs in FLOWSHEETS]
        ax.bar(
            _xs,
            heights,
            bottom=bottoms,
            label=cat,
            color=_CAT_COLORS[cat],
            edgecolor="white",
            linewidth=0.6,
            width=_BAR_W,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=10)
    ax.set_ylabel("Annual OPEX [M€/yr]", fontsize=10)
    ax.set_title(
        f"OPEX Breakdown — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_opex_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_lifecycle_breakdown_stacked(mem, psa, amine):
    """Stacked bar: lifecycle cost (CAPEX + PVF×OPEX) by category for each flowsheet."""
    _econ = Economics()
    pvf = _econ.pvf
    op_hours = _econ.operating_hours_per_year

    def _lc_breakdown(m):
        # CAPEX portions
        co2_elec_capex = _safe(m.elec.CAPEX)
        h2_elec_capex = _safe(m.h2_elec.CAPEX) if hasattr(m, "h2_elec") else 0.0
        sep_capex = 0.0
        for attr in ("comp", "mem", "psa", "amine"):
            try:
                sub = getattr(m, attr, None)
                if sub is not None:
                    sep_capex += _safe(sub.total_capex if hasattr(sub, "total_capex") else sub.CAPEX)
            except Exception:
                pass
        if hasattr(m, "flash_CAPEX"):
            sep_capex += _safe(m.flash_CAPEX)

        # OPEX portions [€/yr]
        co2_elec_elec = _safe(m.elec.Power) * _econ.price_power / 1000 * op_hours
        co2_elec_maint = _safe(m.elec.maint_factor) * _safe(m.elec.CAPEX)
        h2_elec_elec = _safe(m.h2_elec.OPEX) if hasattr(m, "h2_elec") else 0.0
        h2_elec_maint = _safe(m.h2_elec.MAINTENANCE) if hasattr(m, "h2_elec") else 0.0
        feedstock = _safe(m.cost_CO2_feed) if hasattr(m, "cost_CO2_feed") else 0.0
        sep_opex = _safe(m.total_opex) - co2_elec_elec - co2_elec_maint - h2_elec_elec - h2_elec_maint - feedstock

        return {
            "CO₂ elec. CAPEX": co2_elec_capex,
            "H₂ elec. CAPEX": h2_elec_capex,
            "Separation CAPEX": sep_capex,
            "CO₂ elec. electricity": pvf * co2_elec_elec,
            "CO₂ elec. maintenance": pvf * co2_elec_maint,
            "H₂ elec. electricity": pvf * h2_elec_elec,
            "H₂ elec. maintenance": pvf * h2_elec_maint,
            "Separation OPEX": pvf * sep_opex,
            "Feedstock (CO₂)": pvf * feedstock,
        }

    breakdowns = {
        "Membrane": _lc_breakdown(mem),
        "PSA": _lc_breakdown(psa),
        "Amine": _lc_breakdown(amine),
    }

    all_cats = [
        "CO₂ elec. CAPEX",
        "H₂ elec. CAPEX",
        "Separation CAPEX",
        "CO₂ elec. electricity",
        "CO₂ elec. maintenance",
        "H₂ elec. electricity",
        "H₂ elec. maintenance",
        "Separation OPEX",
        "Feedstock (CO₂)",
    ]

    _CAT_COLORS = {
        "CO₂ elec. CAPEX": "#2166ac",
        "H₂ elec. CAPEX": "#d6604d",
        "Separation CAPEX": "#4dac26",
        "CO₂ elec. electricity": "#6baed6",
        "CO₂ elec. maintenance": "#bdd7e7",
        "H₂ elec. electricity": "#f4a582",
        "H₂ elec. maintenance": "#fddbc7",
        "Separation OPEX": "#a1d99b",
        "Feedstock (CO₂)": "#b8b8b8",
    }

    _BAR_W = 0.3
    _xs = [i * _BAR_W * 2 for i in range(len(FLOWSHEETS))]
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0.0] * len(FLOWSHEETS)

    for cat in all_cats:
        heights = [breakdowns[fs].get(cat, 0.0) / 1e6 for fs in FLOWSHEETS]
        ax.bar(
            _xs,
            heights,
            bottom=bottoms,
            label=cat,
            color=_CAT_COLORS[cat],
            edgecolor="white",
            linewidth=0.6,
            width=_BAR_W,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=10)
    ax.set_ylabel(f"Lifecycle Cost [M€]  (CAPEX + PVF×OPEX, PVF={pvf:.2f})", fontsize=9)
    ax.set_title(
        f"Lifecycle Cost Breakdown — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_lifecycle_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # CSV of the plotted data (lifecycle cost by category, M€)
    import csv

    out_csv = _PLOTS_DIR / "comparison_lifecycle_breakdown.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category [M€]"] + FLOWSHEETS)
        for cat in all_cats:
            writer.writerow([cat] + [breakdowns[fs].get(cat, 0.0) / 1e6 for fs in FLOWSHEETS])
        writer.writerow(["Total"] + [sum(breakdowns[fs].values()) / 1e6 for fs in FLOWSHEETS])
    print(f"Saved: {out_csv}")


# ---------------------------------------------------------------------------
# Electrolyzer operating point: lambda & current density
# ---------------------------------------------------------------------------


def plot_lambda_current_density(mem, psa, amine):
    """Grouped twin-axis bar chart of the CO2 electrolyzer operating point
    (CO2 stoichiometric ratio λ and current density) for each flowsheet."""
    models = [("Membrane", mem), ("PSA", psa), ("Amine", amine)]

    lam_vals, cd_vals = [], []
    for _, m in models:
        e = m.elec
        e.calculate_performance_metrics()  # ensure λ / CD are up to date
        lam_vals.append(_safe(e.lamda_CO2))
        cd_vals.append(_safe(e.CD))

    _C_LAM = "#4C72B0"
    _C_CD = "#DD8452"
    _W = 0.36
    _xs = list(range(len(FLOWSHEETS)))

    fig, ax_lam = plt.subplots(figsize=(7, 5))
    ax_cd = ax_lam.twinx()

    bars_lam = ax_lam.bar(
        [x - _W / 2 for x in _xs],
        lam_vals,
        width=_W,
        color=_C_LAM,
        edgecolor="white",
        linewidth=0.8,
        label="Stoich. CO₂ ratio λ",
    )
    bars_cd = ax_cd.bar(
        [x + _W / 2 for x in _xs],
        cd_vals,
        width=_W,
        color=_C_CD,
        edgecolor="white",
        linewidth=0.8,
        label="Current density",
    )

    for bar, v in zip(bars_lam, lam_vals):
        if v == v:
            ax_lam.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{round(v, 1):g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_C_LAM,
            )
    for bar, v in zip(bars_cd, cd_vals):
        if v == v:
            ax_cd.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{round(v / 10) * 10:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_C_CD,
            )

    ax_lam.set_xticks(_xs)
    ax_lam.set_xticklabels(FLOWSHEETS, fontsize=11)
    ax_lam.set_ylabel("CO₂ stoichiometric ratio λ [-]", fontsize=10, color="black")
    ax_cd.set_ylabel("Current density [A/m²]", fontsize=10, color="black")
    ax_lam.tick_params(axis="y", labelcolor=_C_LAM)
    ax_cd.tick_params(axis="y", labelcolor=_C_CD)
    ax_lam.set_ylim(bottom=0)
    ax_cd.set_ylim(bottom=0)
    # headroom so value labels don't clip
    ax_lam.set_ylim(top=max(lam_vals) * 1.15)
    ax_cd.set_ylim(top=max(cd_vals) * 1.15)

    ax_lam.set_title(
        f"CO₂ Electrolyzer Operating Point — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax_lam.spines[["top"]].set_visible(False)
    ax_cd.spines[["top"]].set_visible(False)

    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_lambda_current_density.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Electrolyzer inlet composition
# ---------------------------------------------------------------------------


def plot_electrolyzer_inlet(mem, psa, amine):
    """Grouped bar chart of the molar flow into the CO2 electrolyzer, split
    into CO2 / CO / H2 [mol/s], for each flowsheet."""
    models = [("Membrane", mem), ("PSA", psa), ("Amine", amine)]

    # species label -> (electrolyzer inlet attribute, bar colour)
    species = [
        ("CO₂", "nf_CO2_elec_in", "#2166ac"),
        ("CO", "nf_CO_elec_in", "#DD8452"),
        ("H₂", "nf_H2_elec_in", "#55A868"),
    ]

    data = {lbl: [_safe(getattr(m.elec, attr)) for _, m in models] for lbl, attr, _ in species}

    n_sp = len(species)
    _W = 0.25
    _xs = list(range(len(FLOWSHEETS)))
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (lbl, _, color) in enumerate(species):
        offsets = [x + (i - (n_sp - 1) / 2) * _W for x in _xs]
        bars = ax.bar(offsets, data[lbl], width=_W, label=lbl, color=color, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, data[lbl]):
            if v == v:  # not NaN
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=11)
    ax.set_ylabel("Molar flow into electrolyzer [mol/s]", fontsize=10)
    ax.set_title(
        f"Electrolyzer Inlet Composition — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_electrolyzer_inlet.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Electrolyzer CO2 conversion (heterogeneous / homogeneous / total)
# ---------------------------------------------------------------------------


def _curly_brace_v(ax, ylow, yhigh, x0, width):
    """Draw a vertical right-facing curly brace spanning ylow..yhigh, with its
    left edge at x0 and tip bulging right by `width` (x data units).
    Returns the tip x-coordinate (for placing a label just to its right)."""
    import numpy as np

    yax0, yax1 = ax.get_ylim()
    yspan = (yax1 - yax0) or 1.0
    res = max(int(abs(yhigh - ylow) / yspan * 100) * 2 + 1, 11)
    beta = 300.0 / yspan
    y = np.linspace(ylow, yhigh, res)
    y_half = y[: res // 2 + 1]
    x_half = 1.0 / (1.0 + np.exp(-beta * (y_half - y_half[0]))) + 1.0 / (1.0 + np.exp(-beta * (y_half - y_half[-1])))
    x = np.concatenate((x_half, x_half[-2::-1]))
    x = x0 + (x - x.min()) / (x.max() - x.min()) * width  # normalise to [x0, x0+width]
    ax.plot(x, y, color="black", lw=1.0, clip_on=False)
    return x0 + width


def plot_conversion(mem, psa, amine):
    """Stacked bar chart of CO2 conversion in the electrolyzer — heterogeneous
    (electrochemical CO production) plus homogeneous (side reactions) stacked
    to the total [%] — one bar per flowsheet.  Note X_tot = X_het + X_hom."""
    models = [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
    for _, m in models:
        m.elec.calculate_performance_metrics()  # populate X_het / X_hom / X_tot

    # stacked segments, bottom -> top: (label, attribute, colour)
    segments = [
        ("Heterogeneous conversion", "X_het", "#DD8452"),
        ("Homogeneous consumption", "X_hom", "#8172B3"),
    ]
    tot = [_safe(m.elec.X_tot) for _, m in models]

    _W = 0.25
    _GAP = 0.55  # spacing between bars (smaller = closer)
    _xs = [i * _GAP for i in range(len(FLOWSHEETS))]
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw stacked bars; remember each segment's (ylow, yhigh, value) per bar.
    seg_geom = {j: [] for j in range(len(FLOWSHEETS))}
    bottoms = [0.0] * len(FLOWSHEETS)
    for lbl, attr, color in segments:
        heights = [_safe(getattr(m.elec, attr)) for _, m in models]
        ax.bar(_xs, heights, width=_W, bottom=bottoms, label=lbl, color=color, edgecolor="white", linewidth=0.8)
        for j, (b, h) in enumerate(zip(bottoms, heights)):
            if h == h:
                seg_geom[j].append((b, b + h, h))
        bottoms = [b + (h if h == h else 0.0) for b, h in zip(bottoms, heights)]

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=11)
    ax.set_ylabel("CO₂ conversion [%]", fontsize=10)
    ax.set_title(
        f"CO₂ Electrolyzer Conversion — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-_GAP, _xs[-1] + _GAP)
    ax.set_ylim(bottom=0, top=max([t for t in tot if t == t], default=1) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    # Curly brace + value to the right of each stacked segment.
    _brace_w = 0.03  # horizontal bulge of the brace, in x data units
    for i, x in enumerate(_xs):
        bar_right = x + _W / 2
        for ylow, yhigh, v in seg_geom[i]:
            tip = _curly_brace_v(ax, ylow, yhigh, bar_right + 0.015, _brace_w)
            ax.text(tip + 0.01, (ylow + yhigh) / 2, f"{v:.0f}", ha="left", va="center", fontsize=8)

    # Total on top of each stack (regular weight).
    for x, t in zip(_xs, tot):
        if t == t:
            ax.text(x, t, f"{t:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_conversion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(mem: MembraneFlowsheet, psa: PSAFlowsheet, amine: AmineWashFlowsheet):
    import csv

    def _mf_recycle_co2(m) -> float:
        """CO2 molar flow in the recycle stream [mol/s]."""
        pf = 1.0 - _safe(m.purge_fraction)
        if hasattr(m, "mem"):
            return pf * _safe(m.mem.nf_p_CO2_out)
        if hasattr(m, "psa"):
            return _safe(m.psa.nf_CO2_psaL_out)
        if hasattr(m, "amine"):
            return _safe(m.amine.nf_CO2_captured)
        return float("nan")

    def _mf_recycle_co(m) -> float:
        """CO molar flow in the recycle stream [mol/s]."""
        pf = 1.0 - _safe(m.purge_fraction)
        if hasattr(m, "mem"):
            return pf * _safe(m.mem.nf_p_CO_out)
        if hasattr(m, "psa"):
            return _safe(m.psa.nf_CO_psaL_out)
        if hasattr(m, "amine"):
            return 0.0  # amine captures CO2 only; CO passes straight to syngas
        return float("nan")

    def _sep_y_CO2_in(m) -> float:
        """CO2 mole fraction in the gas fed to the separation unit.
        Membrane: full CO2+CO+H2 feed; PSA: CO2+CO only (H2 bypasses);
        Amine: model's absorber inlet y_CO2_in."""
        if hasattr(m, "mem"):
            c = _safe(m.mem.nf_r_CO2[1])
            co = _safe(m.mem.nf_r_CO[1])
            h = _safe(m.mem.nf_r_H2[1])
            tot = c + co + h
            return c / tot if tot else float("nan")
        if hasattr(m, "psa"):
            c = _safe(m.psa.nf_CO2_psaH_in)
            co = _safe(m.psa.nf_CO_psaH_in)
            tot = c + co
            return c / tot if tot else float("nan")
        if hasattr(m, "amine"):
            return _safe(m.amine.y_CO2_in)
        return float("nan")

    def _amine_opex_shares(m) -> dict:
        """Cost-component shares of the amine separation-unit OPEX [%].
        Empty strings for non-amine flowsheets."""
        keys = ("steam", "maint", "mea", "cool", "elec")
        if not hasattr(m, "amine"):
            return {k: "" for k in keys}
        a = m.amine
        kW2GJh = 3.6e-3  # GJ/(kW·h)
        steam = _safe(a.Q_reboiler_kW) * kW2GJh * _safe(a.steam_cost)
        elec = _safe(a.W_elec_kW) * _safe(a.price_power)
        cool = _safe(a.Q_cooling_kW) * kW2GJh * _safe(a.cooling_cost)
        mea = _safe(a.mea_makeup_cost) * _safe(a.scale_tph)
        maint = _safe(a.maintenance_factor) * _safe(a.CAPEX) / _safe(a.op_hours)
        opex = _safe(a.OPEX)
        den = opex if opex else float("nan")
        return {
            "steam": steam / den * 100,
            "maint": maint / den * 100,
            "mea": mea / den * 100,
            "cool": cool / den * 100,
            "elec": elec / den * 100,
        }

    def _row(m, label):
        e = m.elec
        e.calculate_performance_metrics()  # ensures FE is up-to-date for all flowsheets
        _shares = _amine_opex_shares(m)
        return {
            # --- Identity ---
            "Flowsheet": label,
            "H2_type": H2_TYPE,
            "Syngas target [t/day]": SYNGAS_TARGET_TPD,
            # --- Syngas product ---
            "Syngas output [t/day]": _safe(m.mf_syngas) * 86400 / 1000,
            "H2/CO ratio [-]": _safe(m.nf_H2_syngas) / max(_safe(m.nf_CO_syngas), 1e-12),
            "nf_CO_syngas [mol/s]": _safe(m.nf_CO_syngas),
            "nf_H2_syngas [mol/s]": _safe(m.nf_H2_syngas),
            "nf_CO2_syngas [mol/s]": _safe(m.nf_CO2_syngas),
            "Syngas purity (H2+CO) [-]": (_safe(m.nf_H2_syngas) + _safe(m.nf_CO_syngas))
            / max(_safe(m.nf_H2_syngas) + _safe(m.nf_CO_syngas) + _safe(m.nf_CO2_syngas), 1e-12),
            # --- CO2 feed & recycle ---
            "n_CO2_fresh [mol/s]": _safe(m.n_CO2_fresh),
            "CO2 feed intensity [t/t syngas]": _safe(m.n_CO2_fresh)
            * 44e-3
            * 86400
            / 1000
            / max(_safe(m.mf_syngas) * 86400 / 1000, 1e-12),
            "nf_CO2_recycle [mol/s]": _mf_recycle_co2(m),
            "nf_CO_recycle [mol/s]": _mf_recycle_co(m),
            "nf_H2_buy [mol/s]": _safe(m.nf_H2_buy) if hasattr(m, "nf_H2_buy") else 0.0,
            "sep. CO2 inlet mole fraction [-]": _sep_y_CO2_in(m),
            # --- CO2 electrolyzer outputs ---
            "elec.nf_CO2_out [mol/s]": _safe(e.nf_CO2_elec_out),
            "elec.nf_CO_out [mol/s]": _safe(e.nf_CO_elec_out),
            "elec.nf_H2_out [mol/s]": _safe(e.nf_H2_elec_out),
            "elec.CO2share [-]": _safe(e.CO2share),
            "elec.lamda_CO2 [-]": _safe(e.lamda_CO2),
            # --- CO2 electrolyzer operating point ---
            "elec.CD [A/m2]": _safe(e.CD),
            "elec.Vcell [V]": _safe(e.Vcell),
            "elec.Power [MW]": _safe(e.Power) / 1e6,
            "elec.total_area [m2]": _safe(e.total_area),
            "elec.n_cells [-]": _safe(e.n),
            "elec.FE_CO [%]": _safe(e.FE),
            "elec.FE_H2 [%]": 100.0 - _safe(e.FE),
            # --- H2 electrolyzer ---
            "h2_elec.power [MW]": _safe(m.h2_elec.power_hp_mw) if hasattr(m, "h2_elec") else 0.0,
            # --- Economics ---
            "CAPEX [M€]": _safe(m.total_capex) / 1e6,
            "OPEX [M€/yr]": _safe(m.total_opex) / 1e6,
            "Lifecycle cost [M€]": _safe(m.total_lifecycle_cost) / 1e6,
            # --- Membrane-specific ---
            "mem.compressor_power [kW]": (
                sum(_safe(c.power_kw) for c in m.comp._comps) if hasattr(m, "comp") else ""
            ),
            "mem.A_mem [m2]": _safe(m.mem.A_mem) if hasattr(m, "mem") else "",
            "mem.stage_cut [-]": _safe(m.mem.stage_cut) if hasattr(m, "mem") else "",
            "mem.p_retentate [bar]": _safe(m.mem.p_r) / 1e5 if hasattr(m, "mem") else "",
            "mem.p_permeate [bar]": _safe(m.mem.p_p) / 1e5 if hasattr(m, "mem") else "",
            # --- PSA-specific ---
            "psa.CO_recovery [-]": (
                _safe(m.psa.nf_CO_psaH_out) / max(_safe(m.psa.nf_CO_psaH_in), 1e-12) if hasattr(m, "psa") else ""
            ),
            "psa.p_H [kPa]": _safe(m.psa.p_H) if hasattr(m, "psa") else "",
            "psa.p_L [kPa]": _safe(m.psa.p_L) if hasattr(m, "psa") else "",
            "psa.m_ads [kg]": _safe(m.psa.m_ads) if hasattr(m, "psa") else "",
            "psa.power_comp [kW]": _safe(m.psa.power_comp_in) if hasattr(m, "psa") else "",
            "psa.power_pump [kW]": _safe(m.psa.power_pump_in) if hasattr(m, "psa") else "",
            # --- Amine-specific ---
            "amine.y_CO2_in [-]": _safe(m.amine.y_CO2_in) if hasattr(m, "amine") else "",
            "amine.capture_rate [-]": _safe(m.amine.capture_rate) if hasattr(m, "amine") else "",
            "amine.Q_reboiler [GJ/tCO2]": _safe(m.amine.Q_reboiler_spec) if hasattr(m, "amine") else "",
            "amine.W_elec [kWh/tCO2]": _safe(m.amine.W_elec_spec) if hasattr(m, "amine") else "",
            "amine.Q_cooling [GJ/tCO2]": _safe(m.amine.Q_cooling_spec) if hasattr(m, "amine") else "",
            "amine.scale [tCO2/h]": _safe(m.amine.scale_tph) if hasattr(m, "amine") else "",
            # --- Amine OPEX cost-component shares [% of amine separation OPEX] ---
            "amine.OPEX steam share [%]": _shares["steam"],
            "amine.OPEX maintenance share [%]": _shares["maint"],
            "amine.OPEX MEA makeup share [%]": _shares["mea"],
            "amine.OPEX cooling share [%]": _shares["cool"],
            "amine.OPEX electricity share [%]": _shares["elec"],
        }

    rows = [_row(mem, "Membrane"), _row(psa, "PSA"), _row(amine, "Amine")]
    out = _PLOTS_DIR / "operating_points.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Variable"] + [r["Flowsheet"] for r in rows])
        for key in rows[0].keys():
            if key == "Flowsheet":
                continue
            writer.writerow([key] + [r[key] for r in rows])
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Production cost per tonne syngas
# ---------------------------------------------------------------------------


def plot_production_cost(mem, psa, amine):
    """Stacked bar: annualised CAPEX + OPEX per tonne of syngas produced."""
    _econ = Economics()
    plant_life = _econ.plant_life_years

    def _costs_per_tonne(m):
        annual_t = _safe(m.mf_syngas) * 86400 / 1000 * 365  # t/yr
        lifetime_t = annual_t * plant_life  # t over plant life
        capex_pt = _safe(m.total_capex) / lifetime_t  # €/t
        opex_pt = _econ.pvf * _safe(m.total_opex) / lifetime_t  # €/t (discounted)
        return capex_pt, opex_pt

    models = [("Membrane", mem), ("PSA", psa), ("Amine", amine)]
    capex_vals = []
    opex_vals = []
    for _, m in models:
        c, o = _costs_per_tonne(m)
        capex_vals.append(c)
        opex_vals.append(o)

    _BAR_W = 0.4
    _xs = range(len(FLOWSHEETS))
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(_xs, capex_vals, width=_BAR_W, label="Annualised CAPEX", color="#2166ac", edgecolor="white", linewidth=0.8)
    ax.bar(
        _xs,
        opex_vals,
        width=_BAR_W,
        label="Annual OPEX",
        bottom=capex_vals,
        color="#d6604d",
        edgecolor="white",
        linewidth=0.8,
    )

    # Label total on top of each bar
    for x, c, o in zip(_xs, capex_vals, opex_vals):
        total = c + o
        ax.text(x, total * 1.01, f"{round(total / 10) * 10:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(_xs)
    ax.set_xticklabels(FLOWSHEETS, fontsize=11)
    ax.set_ylabel("Production cost [€/t syngas]", fontsize=10)
    ax.set_title(
        f"Production Cost — {SYNGAS_TARGET_TPD:.0f} t/day syngas{_subtitle()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = _PLOTS_DIR / "comparison_production_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print(f"\nRunning all flowsheets at {SYNGAS_TARGET_TPD} t/day | H2: {H2_TYPE}")

    mem = build_membrane()
    psa = build_psa()
    amine = build_amine()

    metrics = extract_metrics(mem, psa, amine)
    print_comparison(metrics)

    plot_cost_comparison(metrics)
    plot_energy_comparison(metrics)
    plot_process_comparison(metrics)
    plot_cost_breakdown_stacked(mem, psa, amine)
    plot_opex_breakdown_stacked(mem, psa, amine)
    plot_lifecycle_breakdown_stacked(mem, psa, amine)
    plot_production_cost(mem, psa, amine)
    plot_lambda_current_density(mem, psa, amine)
    plot_electrolyzer_inlet(mem, psa, amine)
    plot_conversion(mem, psa, amine)
    export_csv(mem, psa, amine)

    print("\nDone.")


if __name__ == "__main__":
    main()
