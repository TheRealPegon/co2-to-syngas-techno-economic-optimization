# Add parent directory to sys.pathimport sys
import os
import sys
import pyomo.environ as pyo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from configuration import Economics, Constants, PSAParameters

# Now import the module

from SubModels.PSAModel import PSA


def test_psa():
    # p_product_out_bar: compress product from p_H to this pressure [bar].
    # Set to None to skip post-compression.
    model = PSA(allow_vacuum=True, p_product_out_bar=25.0)
    # Fix inlet flows (from electrolyzer operating point)
    model.nf_CO2_psaH_in.fix(252)
    model.nf_CO_psaH_in.fix(28.38)

    # Objective: minimise CAPEX + 20 years * OPEX
    # op_cost_psa covers electricity only (€/h). Adsorbent replacement is added
    # separately: annualised cost = price_adsorbent / adsorbent_lifetime_years,
    # scaled over the plant life. This ensures the solver penalises large adsorbent
    # inventories not just through CAPEX but also through recurring replacement cost.
    _op_hours_per_year = 8400
    _plant_life = 20
    _ads_lifetime = PSAParameters.adsorbent_lifetime_years
    model.test_obj = pyo.Objective(
        expr=(model.CAPEX
              + _plant_life * _op_hours_per_year * model.op_cost_psa
              + _plant_life * model.price_adsorbent / _ads_lifetime),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=True, options={"max_iter": 10000})

    print(f"\nSolver status : {result.solver.status}")
    print(f"Solver termination: {result.solver.termination_condition}")
    model.print_model()

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------
    print("\n=== CONSTRAINT VIOLATIONS ===")
    violations = []
    for c in model.component_objects(pyo.Constraint, active=True):
        for idx in c:
            con = c[idx] if idx is not None else c
            try:
                lb   = pyo.value(con.lower) if con.has_lb() else None
                ub   = pyo.value(con.upper) if con.has_ub() else None
                body = pyo.value(con.body)
                viol = 0.0
                if lb is not None and body < lb:
                    viol = lb - body
                if ub is not None and body > ub:
                    viol = body - ub
                if viol > 1e-6:
                    name = f"{c.name}[{idx}]" if idx is not None else c.name
                    violations.append((viol, name, body, lb, ub))
            except Exception:
                pass
    if violations:
        violations.sort(reverse=True)
        for viol, name, body, lb, ub in violations:
            print(f"  {name}: violation={viol:.4g}  body={body:.6g}  lb={lb}  ub={ub}")
    else:
        print("  All constraints satisfied within 1e-6")

    print("\n=== VARIABLES AT BOUNDS ===")
    for v in model.component_objects(pyo.Var, active=True):
        for idx in v:
            var = v[idx] if idx is not None else v
            try:
                val = pyo.value(var)
                lb, ub = var.lb, var.ub
                name = f"{v.name}[{idx}]" if idx is not None else v.name
                if lb is not None and abs(val - lb) < 1e-4 * max(abs(lb), 1):
                    print(f"  AT_LB  {name} = {val:.6g}  [{lb}, {ub}]")
                elif ub is not None and abs(val - ub) < 1e-4 * max(abs(ub), 1):
                    print(f"  AT_UB  {name} = {val:.6g}  [{lb}, {ub}]")
            except Exception:
                pass


if __name__ == "__main__":

    test_psa()
