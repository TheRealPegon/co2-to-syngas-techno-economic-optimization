import os
import sys
import pyomo.environ as pyo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.GasPermeationMembraneModel import GasPermeationMembrane
from SubModels.CompressionFlash.MultiStageCompressionFlowsheet import MultiStageCompressionFlowsheet
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer

def test_membrane_pebax_basic():
    """Solve PEBAX membrane with 100 mol/s feed (60% CO2, 30% CO, 10% H2) and check mole balance."""
    model = GasPermeationMembrane(N_segments=5, membrane_material="PEBAX", co2_depletion_target=98.0)

    model.nf_r_CO2[1].fix(60.0)  # mol/s
    model.nf_r_CO[1].fix(30.0)  # mol/s
    model.nf_r_H2[1].fix(10.0)  # mol/s
    model.p_r.fix(10e5)  # Pa (10 bar)
    model.p_p.fix(1e5)  # Pa (1 bar)
    # A_mem is left free — solver finds the area needed to meet the depletion target

    model.test_obj = pyo.Objective(expr=0)

    solver = pyo.SolverFactory("ipopt")
    results = solver.solve(model, tee=True)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    N = model._N
    feed_total = 100.0
    ret_total = pyo.value(model.nf_r_CO2[N + 1] + model.nf_r_CO[N + 1] + model.nf_r_H2[N + 1])
    perm_total = pyo.value(model.nf_p_CO2_out + model.nf_p_CO_out + model.nf_p_H2_out)

    assert (
        abs(feed_total - ret_total - perm_total) < 1e-3
    ), f"Mole balance violated: feed={feed_total:.4f}, ret={ret_total:.4f}, perm={perm_total:.4f}"

    model.print_variables()


def test_pebax_with_compression():
    """2-stage compression (1→10 bar) followed by PEBAX membrane with 98% CO2 depletion target."""
    z_dict    = {"CO2": 0.6, "CO": 0.3, "H2": 0.1}
    feed_flow = 100.0  # mol/s

    # --- Compression ---
    comp_model = MultiStageCompressionFlowsheet(
        z_dict=z_dict,
        feed_flow=feed_flow,
        n_stages=2,
        Pin=1.0,
        Pout=10.0,
    )
    solver = pyo.SolverFactory("ipopt")
    comp_results = solver.solve(comp_model, tee=False)
    assert comp_results.solver.termination_condition == pyo.TerminationCondition.optimal
    comp_model.report()

    # --- Membrane: connect compression outlet → membrane feed ---
    # Compression does not change composition; outlet pressure drives membrane p_r.
    p_out_bar   = pyo.value(comp_model.Pout_var) if hasattr(comp_model, "Pout_var") \
                  else pyo.value(comp_model._comps[-1].Pout)
    total_flow  = pyo.value(comp_model._comps[0].molar_flow)

    mem_model = GasPermeationMembrane(N_segments=5, membrane_material="PEBAX", co2_depletion_target=98.0)

    mem_model.nf_r_CO2[1].fix(total_flow * z_dict["CO2"])
    mem_model.nf_r_CO[1].fix(total_flow * z_dict["CO"])
    mem_model.nf_r_H2[1].fix(total_flow * z_dict["H2"])
    mem_model.p_r.fix(p_out_bar * 1e5)  # Pa
    mem_model.p_p.fix(1e5)  # Pa

    mem_model.test_obj = pyo.Objective(expr=0)
    mem_results = solver.solve(mem_model, tee=False)
    assert mem_results.solver.termination_condition == pyo.TerminationCondition.optimal

    N = mem_model._N
    feed_total = feed_flow
    ret_total  = pyo.value(mem_model.nf_r_CO2[N+1] + mem_model.nf_r_CO[N+1] + mem_model.nf_r_H2[N+1])
    perm_total = pyo.value(mem_model.nf_p_CO2_out + mem_model.nf_p_CO_out + mem_model.nf_p_H2_out)
    assert abs(feed_total - ret_total - perm_total) < 1e-3, \
        f"Mole balance violated: feed={feed_total:.4f}, ret={ret_total:.4f}, perm={perm_total:.4f}"

    mem_model.print_variables()


def test_electrolyzer_compression_membrane():
    """Electrolyzer (CO2share=99%, no reflow) → 2-stage compression → PEBAX membrane."""
    solver = pyo.SolverFactory("ipopt")

    # --- Electrolyzer ---
    # Fix: CO2share=99%, nf_CO_elec_out=30 mol/s; let E_appl, v, and area float
    elec = CO2Electrolyzer()
    elec.CO2share.fix(0.99)
    elec.nf_CO_elec_out.fix(30.0)

    # Seed initial values: inlet concentrations follow from CO2share via ideal gas
    # c = (P/RT) * share;  P=101325 Pa, T=298.15 K, R=8.314 J/mol·K  → ~40.9 mol/m³ total
    _c_total = 101325 / (8.314 * 298.15)
    elec.c_elec_in_CO2.set_value(_c_total * 0.99)
    elec.c_elec_in_CO.set_value(_c_total * 0.01)
    elec.c_elec_out_CO2.set_value(_c_total * 0.5)
    elec.c_elec_out_CO.set_value(1.5)
    elec.c_elec_out_H2.set_value(0.5)
    elec.c_elec_mean_CO2.set_value(_c_total * 0.7 * 0.85)
    # vol_gas_flow seed: nf_CO_out / c_out_CO ≈ 30 / 1.5 = 20 m³/s
    elec.vol_gas_flow.set_value(20.0)
    elec.n.set_value(20.0 / (0.08 * elec.Channel_H * elec.Channel_W * 1e6))
    elec.nf_CO2_elec_out.set_value(_c_total * 0.5 * 20.0)
    elec.nf_H2_elec_out.set_value(0.5 * 20.0)

    elec.test_obj = pyo.Objective(expr=0)
    elec_results = solver.solve(elec, tee=False)
    assert elec_results.solver.termination_condition == pyo.TerminationCondition.optimal
    elec.print_variables()

    nf_CO2 = pyo.value(elec.nf_CO2_elec_out)
    nf_CO  = pyo.value(elec.nf_CO_elec_out)
    nf_H2  = pyo.value(elec.nf_H2_elec_out)
    total  = nf_CO2 + nf_CO + nf_H2
    z_dict = {"CO2": nf_CO2 / total, "CO": nf_CO / total, "H2": nf_H2 / total}
    print(f"\n[Electrolyzer outlet]  CO2={nf_CO2:.3f}  CO={nf_CO:.3f}  H2={nf_H2:.3f}  mol/s")

    # --- Compression ---
    comp = MultiStageCompressionFlowsheet(
        z_dict=z_dict,
        feed_flow=total,
        n_stages=2,
        Pin=1.0,
        Pout=10.0,
    )
    comp_results = solver.solve(comp, tee=False)
    assert comp_results.solver.termination_condition == pyo.TerminationCondition.optimal
    comp.report()

    p_out_bar  = pyo.value(comp.Pout_var) if hasattr(comp, "Pout_var") else pyo.value(comp._comps[-1].Pout)
    feed_flow  = pyo.value(comp._comps[0].molar_flow)

    # --- Membrane ---
    mem = GasPermeationMembrane(N_segments=5, membrane_material="PEBAX", co2_depletion_target=98.0)
    mem.nf_r_CO2[1].fix(feed_flow * z_dict["CO2"])
    mem.nf_r_CO[1].fix(feed_flow * z_dict["CO"])
    mem.nf_r_H2[1].fix(feed_flow * z_dict["H2"])
    mem.p_r.fix(p_out_bar * 1e5)
    mem.p_p.fix(1e5)
    mem.test_obj = pyo.Objective(expr=0)
    mem_results = solver.solve(mem, tee=False)
    assert mem_results.solver.termination_condition == pyo.TerminationCondition.optimal

    N = mem._N
    ret_total  = pyo.value(mem.nf_r_CO2[N+1] + mem.nf_r_CO[N+1] + mem.nf_r_H2[N+1])
    perm_total = pyo.value(mem.nf_p_CO2_out + mem.nf_p_CO_out + mem.nf_p_H2_out)
    assert abs(feed_flow - ret_total - perm_total) < 1e-3, \
        f"Mole balance violated: feed={feed_flow:.4f}, ret={ret_total:.4f}, perm={perm_total:.4f}"
    mem.print_variables()


if __name__ == "__main__":
    test_electrolyzer_compression_membrane()
