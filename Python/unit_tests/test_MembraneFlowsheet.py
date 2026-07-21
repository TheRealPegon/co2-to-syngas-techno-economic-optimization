"""
Tests / demo scripts for CombinedFlowsheet.

Each function demonstrates a different configuration:
  1. No flash, H2 from electrolyzer, no syngas target
  2. No flash, H2 from electrolyzer, syngas target
  3. No flash, H2 bought,            syngas target
  4. With flash, H2 from electrolyzer, syngas target
  5. With flash, H2 bought,            syngas target

Workflow
--------
1. Construct the flowsheet
2. (Optional) call set_syngas_target(...)
3. Call initialize_electrolyzer(...) — sets a self-consistent starting point
   for the coupled recycle loop so IPOPT starts near feasibility
4. Call solve()
5. Call report()

Note: Pylance reports false-positive "not callable" warnings on Pyomo
ConcreteModel subclass method calls. The # type: ignore comments suppress them.
"""

import os
import sys
from typing import cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.MembraneFlowsheet import MembraneFlowsheet  # type: ignore[import]

# Flash equilibrium K-values at ~30 bar / 238 K (-35 °C)
K_FLASH = {"CO2": 0.30, "CO": 50.0, "H2": 200.0}

# Known-good electrolyzer operating point (from test_CO2_Electrolyzer_Surrogate.py)
_E_APPL = -1.31  # V
_V = 0.08  # m/s
_CO2SHARE = 0.80  # CO2/(CO2+CO) at cathode inlet


def _check_results(results, label: str):
    """Print solver status; return True if a solution was loaded."""
    if results is None:
        print(f"\n[{label}] Solver crashed before returning results.")
        return False
    tc = str(results.solver.termination_condition)
    print(f"\n[{label}] Solver status: {tc}")
    return True


def test_flowsheet(purge_fraction: float = 0.10):
    """Compression + membrane, H2 purchased, syngas target 75 t/day.

    Uses a homotopy (continuation) strategy: solve at 40 t/day first,
    then warm-start toward the 75 t/day target in steps.
    """
    purge_fraction = 0.1
    m = cast(
        MembraneFlowsheet,
        MembraneFlowsheet(
            n_stages=1,
            with_flash=False,
            membrane_material="PEBAX",
            purge_fraction=purge_fraction,
            float_purge=False,
            recycle_permeate_H2=False,
        ),
    )
    # Build model with 75 t/day target (sets variable initial guesses accordingly)
    # m.set_syngas_target(syngas_target=75.0, syngas_ratio=2.0, H2_type="Buy")  # type: ignore[operator]
    m.set_syngas_target(syngas_target=75.0, syngas_ratio=2.0, H2_type="H2Electrolyzer")  # type: ignore[operator]
    m.initialize_electrolyzer(E_appl=_E_APPL, v=_V, CO2share=_CO2SHARE)  # type: ignore[operator]

    # Homotopy: lower target first, solve, then ramp up to 75 t/day
    _homotopy_targets = [40.0, 55.0, 65.0, 75.0]  # t/day
    results = None
    for _i, _tgt in enumerate(_homotopy_targets):
        _kgs = _tgt * 1000.0 / 86400.0
        m.mf_syngas_target.set_value(_kgs)  # type: ignore[operator]
        print(f"\n[HOMOTOPY] Solving at syngas target = {_tgt:.1f} t/day ...")
        if _i == 0:
            results = m.solve(show_log=True)  # type: ignore[operator]
        else:
            results = m.solve(  # type: ignore[operator]
                show_log=True,
                solver_options={
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": 1e-9,
                    "warm_start_mult_bound_push": 1e-9,
                    "mu_init": 1e-3,
                },
            )
        _tc = results.solver.termination_condition
        print(f"  -> termination: {_tc}")

    if _check_results(results, "no_flash_buy_H2_target"):
        m.report()  # type: ignore[operator]
    return m


if __name__ == "__main__":
    test_flowsheet()
