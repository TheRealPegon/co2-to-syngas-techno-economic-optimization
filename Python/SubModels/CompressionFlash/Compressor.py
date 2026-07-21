import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pyomo.environ as pyo
from configuration import Economics, PSAParameters


class MolarCompressorModel(pyo.ConcreteModel):
    def __init__(self, name="MolarCompressor", **kwargs):
        super().__init__(name=name)

        # --- Parameters (Inputs) ---
        self.Tin = pyo.Param(initialize=kwargs.get("Tin", 293.15))  # K
        self.Pin = pyo.Param(initialize=kwargs.get("Pin", 1.0))  # bar
        self.Pout = pyo.Param(initialize=kwargs.get("Pout", 30.0))  # bar
        self.gamma = pyo.Param(initialize=kwargs.get("gamma", 1.32))  # Ratio
        self.cp_molar = pyo.Param(initialize=kwargs.get("cp_molar", 35.0))  # J/mol·K
        self.molar_flow = pyo.Param(initialize=kwargs.get("molar_flow", 500.0))  # mol/s
        self.eta = pyo.Param(initialize=kwargs.get("eta", 0.8))

        # Economics — Guthrie correlation (same as PSAModel)
        # base_cost [USD]  = 23000 * (power / 73.55)^0.77
        # F_a_cost  [USD]  = 776.5 * base_cost^0.493
        # installed [USD]  = f_install * (base_cost + F_a_cost)
        # capex_total [EUR]= convert * f_indirect * installed
        _econ = Economics()
        _psa  = PSAParameters()
        self.f_install    = pyo.Param(initialize=_psa.factor_installation_comp)   # 1.565
        self.f_indirect   = pyo.Param(initialize=_psa.indirect_cost_factor_comp)  # 1.38
        self.usd_to_eur   = pyo.Param(initialize=_econ.convert_usd_to_euro)       # 0.96
        self.elec_price   = pyo.Param(initialize=_econ.price_power)               # €/kWh
        self.op_hours     = pyo.Param(initialize=_econ.operating_hours_per_year)  # hr/yr
        # Maintenance: 5 %/yr of installed CAPEX — Peters, Timmerhaus & West, 5th ed. (rotating equipment)
        self.maintenance_factor = pyo.Param(initialize=_econ.maintenance_factor_rotating)
        # CEPCI scaling: Guthrie (1974) → 2024 prices
        self.cepci_ratio = pyo.Param(initialize=_econ.CEPCI_2024 / _econ.CEPCI_1974)

        # --- Variables ---
        self.Tout      = pyo.Var(initialize=400.0)
        self.power_kw  = pyo.Var(initialize=500.0, bounds=(1.0, None))  # lb>0 for pow gradient

        # Guthrie intermediate variables
        self.base_cost_comp      = pyo.Var(initialize=2e5, within=pyo.NonNegativeReals)  # USD
        self.F_a_cost_comp       = pyo.Var(initialize=3e5, within=pyo.NonNegativeReals)  # USD
        self.installed_cost_comp = pyo.Var(initialize=8e5, within=pyo.NonNegativeReals)  # USD
        self.capex_total         = pyo.Var(initialize=1e6, within=pyo.NonNegativeReals)  # EUR

        self.opex_annual = pyo.Var()

        self._set_constraints()
        self.obj = pyo.Objective(expr=0)

    def _set_constraints(self):
        def temp_rule(m):
            return m.Tout == m.Tin + (m.Tin / m.eta) * ((m.Pout / m.Pin) ** ((m.gamma - 1) / m.gamma) - 1)
        self.eq_temperature = pyo.Constraint(rule=temp_rule)

        def power_rule(m):
            return m.power_kw == (m.molar_flow * m.cp_molar * (m.Tout - m.Tin)) / 1000.0
        self.eq_power = pyo.Constraint(rule=power_rule)

        # Guthrie CAPEX — identical structure to PSAModel.py
        # CAPEX basis: actual shaft power (includes η losses; PSA uses isentropic power)
        def base_cost_rule(m):
            return m.base_cost_comp == 23000 * (m.power_kw / 73.55) ** 0.77
        self.eq_base_cost = pyo.Constraint(rule=base_cost_rule)

        def F_a_rule(m):
            return m.F_a_cost_comp == 776.5 * (m.base_cost_comp + 1e-6) ** 0.493
        self.eq_F_a_cost = pyo.Constraint(rule=F_a_rule)

        def installed_rule(m):
            return m.installed_cost_comp == m.f_install * (m.base_cost_comp + m.F_a_cost_comp)
        self.eq_installed_cost = pyo.Constraint(rule=installed_rule)

        def capex_rule(m):
            return m.capex_total == m.usd_to_eur * m.f_indirect * m.installed_cost_comp * m.cepci_ratio
        self.eq_capex = pyo.Constraint(rule=capex_rule)

        def opex_rule(m):
            electricity = m.power_kw * m.op_hours * m.elec_price
            maintenance = m.maintenance_factor * m.capex_total
            return m.opex_annual == electricity + maintenance
        self.eq_opex = pyo.Constraint(rule=opex_rule)

    def solve(self, solver_name="ipopt"):
        return pyo.SolverFactory(solver_name).solve(self)

    def report(self):
        print(f"--- Molar Results for {self.name} ---")
        print(f"Outlet Temp:     {self.Tout.value - 273.15:.2f} °C")
        print(f"Molar Flow:      {self.molar_flow.value:.2f} mol/s")
        print(f"Power:           {self.power_kw.value:.2f} kW")
        print(f"Base cost:       ${self.base_cost_comp.value:,.0f} USD")
        print(f"Installed cost:  ${self.installed_cost_comp.value:,.0f} USD")
        print(f"CAPEX:           €{self.capex_total.value:,.0f}")
        print(f"Annual OPEX:     €{self.opex_annual.value:,.0f}")


if __name__ == "__main__":
    my_comp = MolarCompressorModel(molar_flow=81.4, cp_molar=34.5, gamma=1.32)
    my_comp.solve()
    my_comp.report()
