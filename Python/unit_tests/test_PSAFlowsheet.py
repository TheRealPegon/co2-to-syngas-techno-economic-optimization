import os
import sys
import pyomo.environ as pyo
from pyomo.environ import value

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.PSAFlowsheet import PSAFlowsheet


def test_psa_flowsheet():
    m = PSAFlowsheet(purge_fraction=0.1, float_purge=False)
    m.set_syngas_target(syngas_target=75.0, syngas_ratio=2.0, H2_type="H2Electrolyzer")
    m.initialize()
    results = m.solve(show_log=True)
    m.report()

    assert results is not None, "Solver exited abnormally (ApplicationError)"
    tc = results.solver.termination_condition
    assert tc in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    ), f"Solver terminated with: {tc}"

    # Syngas mass flow within 1 % of target
    target_kgs = 75.0 * 1000.0 / (3600.0 * 24.0)
    assert abs(value(m.mf_syngas) - target_kgs) / target_kgs < 0.01, \
        f"Syngas mass flow {value(m.mf_syngas):.4f} kg/s deviates >1% from target {target_kgs:.4f} kg/s"

    # H2/CO ratio within 1 % of target
    actual_ratio = value(m.nf_H2_syngas) / value(m.nf_CO_syngas)
    assert abs(actual_ratio - 2.0) / 2.0 < 0.01, \
        f"H2/CO ratio {actual_ratio:.4f} deviates >1% from target 2.0"


if __name__ == "__main__":
    test_psa_flowsheet()
