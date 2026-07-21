import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pyomo.environ as pyo
from configuration import HeatXchangerEconomics, Economics, PSAParameters

_COOLING_MEDIA = ("water", "NH3")

# ---------------------------------------------------------------------------
# CAPEX correlation — Guthrie (1969) [ACTIVE]
# Shell-and-tube HX, fixed tube sheet, carbon steel
# C_p = 101.3 × (A_ft²)^0.65   [USD 1969]
# Source: Guthrie, K.M. (1969) "Capital Cost Estimating", Chemical Engineering
# ---------------------------------------------------------------------------
_GUTHRIE_C_REF = 101.3    # USD 1969, reference purchase cost constant — Guthrie (1969)
_GUTHRIE_N     = 0.65     # area exponent (economies of scale)             — Guthrie (1969)
_M2_TO_FT2     = 10.764   # unit conversion: 1 m² = 10.764 ft²
_econ_cls      = Economics()
_CEPCI_RATIO   = _econ_cls.CEPCI_2024 / _econ_cls.CEPCI_1969  # scale from Guthrie 1969 basis to 2024 prices
# Bare module factor for shell-and-tube HX, CS, low pressure — Turton et al. (2012), Table A.4
# F_BM = B1 + B2·Fm·Fp = 1.63 + 1.66·1.0·1.0 = 3.29
_F_BM = 3.29

# ---------------------------------------------------------------------------
# CAPEX correlation — Turton et al. (2012) [INACTIVE — kept as reference]
# "Analysis, Synthesis and Design of Chemical Processes", 4th ed., Table A.1
# Shell-and-tube HX, fixed tube sheet, carbon steel, low pressure
# log10(C_p°) = K1 + K2·log10(A) + K3·(log10(A))²  [USD 2001, A in m², valid 1–1000 m²]
#
# _K1, _K2, _K3    = 4.3247, -0.3030, 0.1634   # Turton (2012), Table A.1
# _LN10             = math.log(10)
# _CEPCI_RATIO_TURTON = 797.9 / 397.0           # 2023 vs 2001 basis
#
# eq_capex_purchased:
#   capex_purchased == exp((_K1 + _K2*log(A)/_LN10 + _K3*(log(A)/_LN10)**2) * _LN10)
# ---------------------------------------------------------------------------


class HeatExchangerModel(pyo.ConcreteModel):
    def __init__(self, name="HeatExchanger", cooling_medium="water", **kwargs):
        """
        Parameters
        ----------
        cooling_medium : str
            "water" or "NH3". Selects U_overall, delta_T_lm, and cooling utility cost.
        Kwargs
        ------
        Tin, Tout, molar_flow, cp_molar  (same as before)
        """
        if cooling_medium not in _COOLING_MEDIA:
            raise ValueError(
                f"cooling_medium must be one of {_COOLING_MEDIA}, got '{cooling_medium}'"
            )
        super().__init__(name=name)

        econ  = HeatXchangerEconomics()
        _econ = Economics()
        _psa  = PSAParameters()

        if cooling_medium == "water":
            U        = econ.U_overall_water    # W/m²K — Perry's, Table 11-10
            dT_lm    = econ.delta_T_lm_water   # K     — engineering assumption
            cool_kwh = econ.cooling_cost_kwh_water
        else:
            U        = econ.U_overall_NH3      # W/m²K — Perry's, Table 11-10
            dT_lm    = econ.delta_T_lm_NH3     # K     — engineering assumption
            cool_kwh = econ.cooling_cost_kwh_NH3

        self.cooling_medium = cooling_medium

        if cooling_medium == "water" and kwargs.get("Tout", 273.15) < 303.15:
            warnings.warn(
                f"Water cooling selected but target outlet temperature "
                f"({kwargs.get('Tout', 273.15) - 273.15:.1f} °C) is below 30 °C. "
                "Consider using NH3 refrigeration instead.",
                UserWarning,
                stacklevel=2,
            )

        # --- Parameters ---
        self.Tin         = pyo.Param(initialize=kwargs.get("Tin", 710.34))
        self.Tout_target = pyo.Param(initialize=kwargs.get("Tout", 273.15))
        self.molar_flow  = pyo.Param(initialize=kwargs.get("molar_flow", 81.4))
        self.cp_molar    = pyo.Param(initialize=kwargs.get("cp_molar", 34.5))

        # UA product [W/m²] → area [m²] via: A = Q[W] / UA
        self.UA = pyo.Param(initialize=U * dT_lm)

        # Guthrie / escalation factors
        self.cepci_ratio = pyo.Param(initialize=_CEPCI_RATIO)
        self.f_indirect  = pyo.Param(initialize=_psa.indirect_cost_factor_comp)  # 1.38
        self.usd_to_eur  = pyo.Param(initialize=_econ.convert_usd_to_euro)       # 0.96

        # OPEX
        self.cooling_cost_kwh = pyo.Param(initialize=cool_kwh)
        self.op_hours         = pyo.Param(initialize=_econ.operating_hours_per_year)
        # Maintenance: 3 %/yr of installed CAPEX — Peters, Timmerhaus & West, 5th ed. (static equipment)
        self.maintenance_factor = pyo.Param(initialize=_econ.maintenance_factor_static)

        # --- Variables ---
        self.duty_kw         = pyo.Var(initialize=500.0)
        self.area_m2         = pyo.Var(initialize=50.0, bounds=(0.1, 5000.0))          # m²
        self.capex_purchased = pyo.Var(initialize=5e4, within=pyo.NonNegativeReals)    # USD 1969
        self.capex_total     = pyo.Var(initialize=2e5, within=pyo.NonNegativeReals)    # EUR
        self.opex_annual     = pyo.Var()

        self._set_constraints()
        self.obj = pyo.Objective(expr=0)

    def _set_constraints(self):
        # 1. Duty from fixed Tin/Tout using cp_molar (deactivated in flowsheets)
        self.eq_duty = pyo.Constraint(
            expr=self.duty_kw
            == (self.molar_flow * self.cp_molar * (self.Tin - self.Tout_target)) / 1000.0
        )

        # 2. Heat transfer area: A [m²] = Q [W] / (U·ΔT_lm)
        self.eq_area = pyo.Constraint(
            expr=self.area_m2 == self.duty_kw * 1000.0 / self.UA
        )

        # 3. Guthrie (1969) purchased cost [USD 1969]:
        #    C_p = 101.3 × (A_ft²)^0.65
        self.eq_capex_purchased = pyo.Constraint(
            expr=self.capex_purchased
            == _GUTHRIE_C_REF * (self.area_m2 * _M2_TO_FT2) ** _GUTHRIE_N
        )

        # 4. Total installed CAPEX [EUR]:
        #    capex_total = C_p × CEPCI_ratio × F_BM × f_indirect × usd_to_eur
        self.eq_capex = pyo.Constraint(
            expr=self.capex_total
            == self.capex_purchased * self.cepci_ratio * _F_BM
            * self.f_indirect * self.usd_to_eur
        )

        # 5. Annual OPEX: cooling utility + maintenance [€/yr]
        self.eq_opex = pyo.Constraint(
            expr=self.opex_annual
            == self.duty_kw * self.op_hours * self.cooling_cost_kwh
            + self.maintenance_factor * self.capex_total
        )

    def solve(self, solver_name="ipopt"):
        return pyo.SolverFactory(solver_name).solve(self)

    def report(self):
        print(f"--- Results for {self.name} ---")
        print(f"Cooling Medium:    {self.cooling_medium}")
        print(f"Inlet Temp:        {pyo.value(self.Tin) - 273.15:.2f} °C")
        print(f"Outlet Temp:       {pyo.value(self.Tout_target) - 273.15:.2f} °C")
        print(f"Cooling Duty:      {pyo.value(self.duty_kw):.2f} kW_th")
        print(f"Heat Exch. Area:   {pyo.value(self.area_m2):.2f} m²")
        print(f"Purchased cost:    ${pyo.value(self.capex_purchased):,.0f} USD (1969)")
        print(f"Total CAPEX:       €{pyo.value(self.capex_total):,.0f}")
        print(f"Annual OPEX:       €{pyo.value(self.opex_annual):,.0f}")


if __name__ == "__main__":
    hx_test = HeatExchangerModel(Tin=710.34, Tout=313.15, molar_flow=81.4)
    hx_test.solve()
    hx_test.report()