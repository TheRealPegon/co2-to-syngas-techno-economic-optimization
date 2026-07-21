"""
Tests / demo scripts for AmineWashFlowsheet.

Each function demonstrates a different H2 sourcing configuration:
  1. H2 from CO2 electrolyzer by-product only
  2. H2 purchased externally
  3. H2 from a dedicated water electrolyzer

Workflow
--------
1. Construct the flowsheet
2. Call set_syngas_target(...)
3. Call initialize_electrolyzer(...) — seeds a self-consistent starting point
4. Call solve()  — using homotopy to step up to the target
5. Call report()

NOTE on y_CO2 bounds
---------------------
The amine wash TEC correlation is validated for 5–50 mol% CO2.
The electrolyzer outlet is typically 40–66 mol% CO2.
The y_CO2_in upper bound of 0.50 forces the optimizer to operate the
electrolyzer at ≥ 50 % CO2 conversion so the outlet fraction stays within
the validated range.  If IPOPT cannot find a feasible point this constraint
is the most likely cause — try a higher purge fraction or a smaller syngas
target.

Note: Pylance reports false-positive "not callable" warnings on Pyomo
ConcreteModel subclass method calls. The # type: ignore comments suppress them.
"""

import os
import sys
from typing import cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.AmineWashFlowsheet import AmineWashFlowsheet  # type: ignore[import]

# Known-good electrolyzer operating point
_E_APPL   = -1.31   # V
_V        = 0.08    # m/s
_CO2SHARE = 0.90    # CO2/(CO2+CO) at cathode inlet
                    # Higher than membrane case — no CO recycled back


def _check_results(results, label: str) -> bool:
    """Print solver status; return True if a solution was loaded."""
    if results is None:
        print(f"\n[{label}] Solver crashed before returning results.")
        return False
    tc = str(results.solver.termination_condition)
    print(f"\n[{label}] Solver status: {tc}")
    return True


def _homotopy_solve(m, targets, label: str, show_log: bool = True):
    """Step through syngas targets and warm-start between steps."""
    results = None
    for _i, _tgt in enumerate(targets):
        _kgs = _tgt * 1000.0 / 86400.0
        m.mf_syngas_target.set_value(_kgs)  # type: ignore[operator]
        print(f"\n[HOMOTOPY {label}] Solving at syngas target = {_tgt:.1f} t/day ...")
        if _i == 0:
            results = m.solve(show_log=show_log)  # type: ignore[operator]
        else:
            results = m.solve(  # type: ignore[operator]
                show_log=show_log,
                solver_options={
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": 1e-9,
                    "warm_start_mult_bound_push": 1e-9,
                    "mu_init": 1e-3,
                },
            )
        tc = results.solver.termination_condition if results else "crashed"
        print(f"  -> termination: {tc}")
    return results


# ---------------------------------------------------------------------------
# Test 1 — H2 from CO2 electrolyzer by-product only
# ---------------------------------------------------------------------------
def test_electrolyzer_h2(syngas_target: float = 75.0):
    """Amine wash flowsheet with H2 sourced from electrolyzer by-product."""
    m = cast(
        AmineWashFlowsheet,
        AmineWashFlowsheet(
            purge_fraction=0.10,
            float_purge=False,
        ),
    )
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=syngas_target,
        syngas_ratio=2.0,
        H2_type="Electrolyzer",
    )
    m.initialize_electrolyzer(  # type: ignore[operator]
        E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE,
    )

    results = _homotopy_solve(m, [40.0, 55.0, 65.0, syngas_target], "elec_H2")

    if _check_results(results, "electrolyzer_H2"):
        m.report()  # type: ignore[operator]
    return m


# ---------------------------------------------------------------------------
# Test 2 — H2 purchased externally
# ---------------------------------------------------------------------------
def test_buy_h2(syngas_target: float = 75.0):
    """Amine wash flowsheet with externally purchased H2."""
    m = cast(
        AmineWashFlowsheet,
        AmineWashFlowsheet(
            purge_fraction=0.10,
            float_purge=False,
        ),
    )
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=syngas_target,
        syngas_ratio=2.0,
        H2_type="Buy",
    )
    m.initialize_electrolyzer(  # type: ignore[operator]
        E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE,
    )

    results = _homotopy_solve(m, [40.0, 55.0, 65.0, syngas_target], "buy_H2")

    if _check_results(results, "buy_H2"):
        m.report()  # type: ignore[operator]
    return m


# ---------------------------------------------------------------------------
# Test 3 — H2 from dedicated water electrolyzer
# ---------------------------------------------------------------------------
def test_h2_electrolyzer(syngas_target: float = 75.0):
    """Amine wash flowsheet with a dedicated H2 water electrolyzer."""
    m = cast(
        AmineWashFlowsheet,
        AmineWashFlowsheet(
            purge_fraction=0.10,
            float_purge=False,
        ),
    )
    m.set_syngas_target(  # type: ignore[operator]
        syngas_target=syngas_target,
        syngas_ratio=2.0,
        H2_type="H2Electrolyzer",
    )
    m.initialize_electrolyzer(  # type: ignore[operator]
        E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE,
    )

    results = _homotopy_solve(m, [40.0, 55.0, 65.0, syngas_target], "H2Elec")

    if _check_results(results, "H2Electrolyzer"):
        m.report()  # type: ignore[operator]
    return m


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_h2_electrolyzer()
