"""
sensitivity_sweep.py
--------------------
Sweeps a single Economics parameter across a range of values and calls
compare_flowsheets to generate the full report (plots + CSV) for each value.

Configuration
-------------
Edit the SWEEP block below to choose which parameter to vary and what values
to test.  Results land in ../Plots/sensitivity_<param>/<value>/.

How parameter patching works
-----------------------------
Economics is a frozen-style dataclass whose __init__ bakes in defaults at
class creation time.  We monkeypatch __init__ to override the target field
after the normal construction, then restore it between runs.
"""

import sys
from pathlib import Path

_HERE        = Path(__file__).resolve().parent   # FullFlowsheets/Scripts/
_PYTHON_ROOT = _HERE.parent.parent.parent        # Python/
sys.path.insert(0, str(_PYTHON_ROOT))
sys.path.insert(0, str(_PYTHON_ROOT / "SubModels"))

import warnings
warnings.filterwarnings("ignore", message="Reassigning the non-component attribute")
warnings.filterwarnings("ignore", category=UserWarning, module="pyomo")

import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")

from configuration import Economics
import compare_flowsheets as cf

# ---------------------------------------------------------------------------
# SWEEP CONFIGURATION — edit here
# ---------------------------------------------------------------------------
# Each entry: (param, values_in_model_units, label, display_values)
_BASE_COST_ELEC = 4670.0  # €/m² — must match ElectrolyzerParameters.base_cost_elec

SWEEPS = [
    (
        "price_CO2_feed",
        [v * 44.01 / 1e6 for v in [0.0, 22.0, 44.0]],  # €/mol
        "CO₂ feed price [€/t CO₂]",
        [0.0, 22.0, 44.0],
    ),
    (
        "price_power",
        [0.04, 0.08, 0.12],  # €/kWh
        "Electricity price [€/kWh]",
        [0.04, 0.08, 0.12],
    ),
    (
        "base_cost_elec",
        [round(_BASE_COST_ELEC * f, 1) for f in [0.5, 1.0, 1.5]],  # €/m²
        "CO₂ electrolyzer cost [€/m²]",
        [round(_BASE_COST_ELEC * f, 1) for f in [0.5, 1.0, 1.5]],
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_economics(param: str, value):
    """Patch `param` on Economics everywhere it is consumed at build time.

    Three layers need patching:
    1. Economics.<param> class attribute — used directly in Pyomo constraint
       expressions in CO2_ElectrolyzerSurrogateModel and PSAModel.
    2. Module-level singleton instances (_E in AmineWash, _econ_cls in
       HeatEXModel) — created once at import time, not rebuilt per run.
    3. Economics.__init__ — so any new Economics() calls during flowsheet
       construction pick up the new value.
    """
    import SubModels.AmineWash as _amine_mod
    import SubModels.CompressionFlash.HeatEXModel as _heatex_mod

    # --- 1. class attribute (used in constraint expressions) ---
    old_class_val = getattr(Economics, param, None)
    setattr(Economics, param, value)

    # --- 2. module-level singleton instances ---
    old_amine_e   = _amine_mod._E.__dict__.get(param)
    old_heatex_e  = _heatex_mod._econ_cls.__dict__.get(param)
    _amine_mod._E.__dict__[param]    = value
    _heatex_mod._econ_cls.__dict__[param] = value

    # --- 3. __init__ for new instances ---
    original_init = Economics.__init__

    def patched_init(self, **kwargs):
        original_init(self, **kwargs)
        self.__dict__[param] = value

    Economics.__init__ = patched_init

    def restore():
        # Restore class attribute
        if old_class_val is None:
            try:
                delattr(Economics, param)
            except AttributeError:
                pass
        else:
            setattr(Economics, param, old_class_val)
        # Restore singleton instances
        if old_amine_e is None:
            _amine_mod._E.__dict__.pop(param, None)
        else:
            _amine_mod._E.__dict__[param] = old_amine_e
        if old_heatex_e is None:
            _heatex_mod._econ_cls.__dict__.pop(param, None)
        else:
            _heatex_mod._econ_cls.__dict__[param] = old_heatex_e
        # Restore __init__
        Economics.__init__ = original_init

    return restore


def _patch_elec_params(param: str, value):
    """Patch a field on ElectrolyzerParameters (class attribute only — no singletons)."""
    from configuration import ElectrolyzerParameters
    old_val = getattr(ElectrolyzerParameters, param, None)
    setattr(ElectrolyzerParameters, param, value)

    def restore():
        if old_val is None:
            try:
                delattr(ElectrolyzerParameters, param)
            except AttributeError:
                pass
        else:
            setattr(ElectrolyzerParameters, param, old_val)

    return restore


# Params that live on ElectrolyzerParameters rather than Economics
_ELEC_PARAM_NAMES = {"base_cost_elec", "f_installation_elec"}


def _patch_param(param: str, value):
    """Route to the correct patch function based on which dataclass owns *param*."""
    if param in _ELEC_PARAM_NAMES:
        return _patch_elec_params(param, value)
    return _patch_economics(param, value)


# ---------------------------------------------------------------------------
# PSA price-homotopy helpers
# ---------------------------------------------------------------------------

def _copy_psa_warmstart(m_from, m_to):
    """Copy all free variable values from a solved PSA flowsheet to a freshly
    built one so IPOPT starts from the previous operating point."""
    import pyomo.environ as pyo
    from pyomo.environ import value as pyo_value

    def _copy_block(blk_from, blk_to):
        for name, comp_from in blk_from.component_map(pyo.Var).items():
            comp_to = blk_to.component(name)
            if comp_to is None:
                continue
            froms = list(comp_from.values()) if comp_from.is_indexed() else [comp_from]
            tos   = list(comp_to.values())   if comp_to.is_indexed()   else [comp_to]
            for v_from, v_to in zip(froms, tos):
                if v_from.is_fixed() or v_to.is_fixed():
                    continue
                try:
                    val = pyo_value(v_from, exception=False)
                    if val is None:
                        continue
                    if v_to.lb is not None:
                        val = max(val, float(v_to.lb) + 1e-10)
                    if v_to.ub is not None:
                        val = min(val, float(v_to.ub) - 1e-10)
                    v_to.set_value(val)
                except Exception:
                    pass
        for name, sub_from in blk_from.component_map(pyo.Block).items():
            sub_to = blk_to.component(name)
            if sub_to is not None:
                _copy_block(sub_from, sub_to)

    _copy_block(m_from, m_to)


def _build_psa_price_homotopy(target_val, from_val, n_intermediate=2):
    """Build and solve the PSA flowsheet by stepping price_CO2_feed from
    *from_val* to *target_val* in (n_intermediate + 1) steps, warm-starting
    each step from the previous solution.

    Called inside the outer _patch_economics context (already patched to
    target_val).  Inner patches temporarily override that to each step value
    and restore cleanly after each step.
    """
    import numpy as np
    from SubModels.PSAFlowsheet import PSAFlowsheet

    # All steps including the final target; exclude from_val itself
    steps = list(np.linspace(from_val, target_val, n_intermediate + 2)[1:])
    disp_from = from_val * 1e6 / 44.01
    disp_to   = target_val * 1e6 / 44.01
    print(f"  [PSA price homotopy: {disp_from:.0f} → {disp_to:.0f} €/t CO₂, "
          f"{len(steps)} step(s)]")

    m_prev = None
    for i, step_val in enumerate(steps):
        restore_step = _patch_param("price_CO2_feed", step_val)
        try:
            if m_prev is None:
                # First intermediate step: cold build (uses patched Economics)
                m_new = cf.build_psa()
            else:
                # Subsequent steps: build fresh, initialize, then override with
                # the previous solution before handing off to IPOPT.
                m_new = PSAFlowsheet(purge_fraction=0.1, float_purge=False)
                m_new.set_syngas_target(
                    syngas_target=cf.SYNGAS_TARGET_TPD,
                    syngas_ratio=cf.SYNGAS_RATIO,
                    H2_type=cf.H2_TYPE,
                    price_CO2_feed=Economics().price_CO2_feed,
                )
                m_new.initialize()
                _copy_psa_warmstart(m_prev, m_new)
                results = m_new.solve(
                    show_log=False,
                    multistart=cf.USE_MULTISTART,
                    multistart_kwargs=cf.MULTISTART_KWARGS,
                )
                tc = results.solver.termination_condition if results else "crashed"
                disp_step = step_val * 1e6 / 44.01
                print(f"  -> PSA homotopy @ {disp_step:.0f} €/t CO₂: {tc}")
        finally:
            restore_step()

        m_prev = m_new

    return m_prev


# ---------------------------------------------------------------------------
# PSA base-case warm-start
# ---------------------------------------------------------------------------
# PSA's cold-start solve is basin-fragile: depending on the ipopt build it
# either reaches the real operating point (CD ~1400-1800 A/m², 75 t/day) or
# stalls in a degenerate low-CD basin (~360-480 A/m², infeasible). We solve it
# ONCE at the base (unpatched) parameters — which converges — and reuse that
# solution to warm-start every sweep point so they all land in the good basin.

_PSA_REF = None  # cached converged base-case PSA, used as warm-start seed


def _psa_reference():
    """Solve PSA once at base (unpatched) parameters; cache as warm-start seed.

    Must be called with no parameter patch active (i.e. before the sweep loop).
    """
    global _PSA_REF
    if _PSA_REF is None:
        print("\n[Building PSA base-case reference for warm-starting]")
        _PSA_REF = cf.build_psa()
    return _PSA_REF


def _build_psa_warmstarted(ref):
    """Build PSA under the currently-patched parameters, warm-started from the
    converged base-case reference solution, then solve."""
    from SubModels.PSAFlowsheet import PSAFlowsheet
    m = PSAFlowsheet(purge_fraction=0.1, float_purge=False)
    m.set_syngas_target(
        syngas_target=cf.SYNGAS_TARGET_TPD,
        syngas_ratio=cf.SYNGAS_RATIO,
        H2_type=cf.H2_TYPE,
        price_CO2_feed=Economics().price_CO2_feed,
    )
    m.initialize()
    if ref is not None:
        _copy_psa_warmstart(ref, m)
    results = m.solve(
        show_log=False,
        multistart=cf.USE_MULTISTART,
        multistart_kwargs=cf.MULTISTART_KWARGS,
    )
    if cf._solver_ok(results):
        print("  -> PSA flowsheet: SOLVED (warm-started from base case)")
    else:
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  -> PSA flowsheet: WARNING — {tc}")
    return m


# ---------------------------------------------------------------------------
# Summary collector
# ---------------------------------------------------------------------------

def _production_cost_per_tonne(m):
    """Return (capex_pt, opex_pt) in €/t for a solved flowsheet."""
    _econ      = Economics()
    plant_life = _econ.plant_life_years
    annual_t   = _s(m.mf_syngas) * 86400 / 1000 * 365
    lifetime_t = annual_t * plant_life
    return (
        _s(m.total_capex)            / lifetime_t,
        _econ.pvf * _s(m.total_opex) / lifetime_t,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def _s(expr):
    from pyomo.environ import value as pyo_value
    try:
        return float(pyo_value(expr))
    except Exception:
        return float("nan")


def _run_sweep(sweep_param, sweep_values, sweep_label, display_values):
    """Run one full parameter sweep and save all plots + summary charts."""
    import matplotlib.pyplot as plt

    summary = {fs: {"capex_pt": [], "opex_pt": [], "cd": [], "total_opex": []}
               for fs in cf.FLOWSHEETS}
    tested_x = []
    csv_rows = []

    # Solve PSA once at base parameters (no patch active here) so every sweep
    # point can be warm-started from a converged operating point.
    psa_ref = _psa_reference()

    for idx, val in enumerate(sweep_values):
        disp = display_values[idx] if display_values else val
        label_val = f"{disp:.4g}".replace(".", "_")
        print(f"\n{'='*60}")
        print(f"  {sweep_label} = {disp}")
        print(f"{'='*60}")

        out_dir = _HERE.parent / "Plots" / f"sensitivity_{sweep_param}" / label_val
        out_dir.mkdir(parents=True, exist_ok=True)
        original_plots_dir = cf._PLOTS_DIR
        cf._PLOTS_DIR = out_dir
        cf.SWEEP_SUBTITLE = f"{sweep_label} = {disp:.4g}"

        restore = _patch_param(sweep_param, val)
        try:
            mem   = cf.build_membrane()

            # Warm-start PSA from the converged base-case solution so it lands
            # in the productive (high-CD) basin regardless of the sweep value
            # or the ipopt build. Replaces the fragile cold cf.build_psa().
            psa = _build_psa_warmstarted(psa_ref)

            amine = cf.build_amine()

            metrics = cf.extract_metrics(mem, psa, amine)
            cf.print_comparison(metrics)
            cf.plot_cost_comparison(metrics)
            cf.plot_energy_comparison(metrics)
            cf.plot_process_comparison(metrics)
            cf.plot_cost_breakdown_stacked(mem, psa, amine)
            cf.plot_opex_breakdown_stacked(mem, psa, amine)
            cf.plot_lifecycle_breakdown_stacked(mem, psa, amine)
            cf.plot_production_cost(mem, psa, amine)
            cf.export_csv(mem, psa, amine)

            tested_x.append(disp)
            costs = {}
            for fs, m in [("Membrane", mem), ("PSA", psa), ("Amine", amine)]:
                c, o = _production_cost_per_tonne(m)
                summary[fs]["capex_pt"].append(c)
                summary[fs]["opex_pt"].append(o)
                summary[fs]["cd"].append(_s(m.elec.CD))
                summary[fs]["total_opex"].append(_s(m.total_opex) / 1e6)
                costs[fs] = c + o
            csv_rows.append({
                "sweep_param":  sweep_param,
                "sweep_label":  sweep_label,
                "display_value": disp,
                "Membrane_cost_eur_per_t": costs["Membrane"],
                "PSA_cost_eur_per_t":      costs["PSA"],
                "Amine_cost_eur_per_t":    costs["Amine"],
            })
        finally:
            restore()
            cf._PLOTS_DIR = original_plots_dir
            cf.SWEEP_SUBTITLE = ""

    if len(tested_x) < 2:
        return csv_rows

    summary_dir = _HERE.parent / "Plots" / f"sensitivity_{sweep_param}"
    colors = {"Membrane": cf.COLORS["Membrane"],
              "PSA":      cf.COLORS["PSA"],
              "Amine":    cf.COLORS["Amine"]}

    def _summary_plot(key, ylabel, title, filename, transform=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for fs in cf.FLOWSHEETS:
            vals = list(summary[fs][key])
            if transform:
                vals = [transform(v) for v in vals]
            ax.plot(tested_x, vals, marker="o", label=fs,
                    color=colors[fs], linewidth=2)
        ax.set_xlabel(sweep_label, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{title}\n({cf.SYNGAS_TARGET_TPD:.0f} t/day syngas, H2: {cf.H2_TYPE})",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(summary_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Summary plot saved: {summary_dir / filename}")
        plt.close(fig)

    _summary_plot("total_opex", "Annual OPEX [M€/yr]",
                  f"Total OPEX vs {sweep_label}", "summary_total_opex.png")
    _summary_plot("cd", "CO₂ electrolyzer current density [A/m²]",
                  f"Current Density vs {sweep_label}", "summary_current_density.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for fs in cf.FLOWSHEETS:
        total_pt = [c + o for c, o in zip(summary[fs]["capex_pt"], summary[fs]["opex_pt"])]
        ax.plot(tested_x, total_pt, marker="o", label=fs, color=colors[fs], linewidth=2)
    ax.set_xlabel(sweep_label, fontsize=11)
    ax.set_ylabel("Production cost [€/t syngas]", fontsize=11)
    ax.set_title(
        f"Production Cost vs {sweep_label}\n"
        f"({cf.SYNGAS_TARGET_TPD:.0f} t/day syngas, H2: {cf.H2_TYPE})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(summary_dir / "summary_production_cost.png", dpi=150, bbox_inches="tight")
    print(f"Summary plot saved: {summary_dir / 'summary_production_cost.png'}")
    plt.close(fig)

    return csv_rows


def main():
    import csv

    all_rows = []
    for sweep_param, sweep_values, sweep_label, display_values in SWEEPS:
        print(f"\n{'#'*60}")
        print(f"  SWEEP: {sweep_label}")
        print(f"{'#'*60}")
        rows = _run_sweep(sweep_param, sweep_values, sweep_label, display_values)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        out_csv = _HERE.parent / "Plots" / "sensitivity_production_costs.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCombined sensitivity CSV saved: {out_csv}")

    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()
