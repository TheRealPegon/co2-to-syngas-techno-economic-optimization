import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pyomo.environ as pyo
from configuration import Economics

# ---------------------------------------------------------------------------
# Turton et al. (2012) "Analysis, Synthesis and Design of Chemical Processes",
# 4th ed., Table A.1 — vertical process vessel, carbon steel
# log10(C_p°) = K1 + K2·log10(V) + K3·(log10(V))²  [USD 2001, V in m³, range 0.3–520 m³]
# ---------------------------------------------------------------------------
_K1_V, _K2_V, _K3_V = 3.4974, 0.4485, 0.1074   # Turton (2012), Table A.1
_LN10 = math.log(10)
# CEPCI 2001 = 397 (Turton basis); CEPCI 2023 = 797.9 (H2Parameters.CE_PCI, configuration.py)
_CEPCI_RATIO = 797.9 / 397.0
# Bare module factor — Turton (2012), Table A.4, vertical vessel
# F_BM = B1 + B2·Fm·Fp;  B1=2.25, B2=1.82
# Fm = 1.0  (carbon steel — CO2/CO/H2 service does not require stainless)
# Fp ≈ 2.0  (30 bar service, D ≈ 0.5 m; from Turton pressure-factor correlation for CS vessels)
_F_BM_VESSEL = 2.25 + 1.82 * 1.0 * 2.0   # = 5.89

# ---------------------------------------------------------------------------
# Previous CAPEX correlation (PSA Guthrie, now inactive — kept for reference):
#   base_cost = 10^6 × (H/1.2192)^0.81 × (D/0.9144)^1.05  [USD, Guthrie 1969]
#   This correlation is calibrated for large industrial adsorption columns and
#   gives ~€7M for a 0.36 m³ vessel — not appropriate for a small flash drum.
# ---------------------------------------------------------------------------


class FlashModel(pyo.ConcreteModel):
    """
    Rachford-Rice flash drum.

    CAPEX: Turton et al. (2012) vertical vessel correlation (Table A.1).
    Sized by vessel volume derived from vapour flow (Souders-Brown criterion).

    OPEX: maintenance only — 3 %/yr of installed CAPEX
          (Peters, Timmerhaus & West, 5th ed., static equipment).
    """

    def __init__(self, z_dict, K_dict, F_total_mols, p, T):
        super().__init__()

        _econ = Economics()
        f_indirect  = 1.38   # indirect cost factor — consistent with PSAParameters.indirect_cost_factor_pv
        usd_to_eur  = _econ.convert_usd_to_euro      # 0.96
        maintenance = _econ.maintenance_factor_static  # 3 %/yr

        # Vessel sizing constants
        rho_v  = 57.5   # kg/m³  — vapour density at flash conditions from aspen
        u_max  = 0.15   # m/s    — maximum allowable vapour velocity (Souders-Brown)
        MW_mix = 0.037  # kg/mol — approximate mixture MW

        # ------------------------------------------------------------------
        # 1. Sets and parameters
        # ------------------------------------------------------------------
        self.I     = pyo.Set(initialize=list(z_dict.keys()))
        self.z     = pyo.Param(self.I, initialize=z_dict)
        self.K     = pyo.Param(self.I, initialize=K_dict)
        self.F     = pyo.Param(initialize=F_total_mols)
        self.P_bar = pyo.Param(initialize=p)

        # ------------------------------------------------------------------
        # 2. Variables
        # ------------------------------------------------------------------
        self.psi = pyo.Var(bounds=(0.001, 0.999), initialize=0.5)

        @self.Expression(self.I)
        def x(self, i):
            return self.z[i] / (1 + self.psi * (self.K[i] - 1))

        @self.Expression(self.I)
        def y(self, i):
            return self.K[i] * self.x[i]

        # Vessel geometry (derived from vapour flow via Souders-Brown criterion)
        @self.Expression()
        def vessel_area(self):
            return (self.F * self.psi * MW_mix / rho_v) / u_max  # m²

        @self.Expression()
        def diameter(self):
            return 2 * pyo.sqrt(self.vessel_area / 3.14159)  # m

        @self.Expression()
        def height(self):
            return self.diameter * 3  # m  (H = 3D rule of thumb)

        @self.Expression()
        def vessel_volume(self):
            return self.vessel_area * self.height  # m³

        self.capex_purchased = pyo.Var(initialize=2e3, within=pyo.NonNegativeReals)  # USD 2001
        self.CAPEX           = pyo.Var(initialize=3e4, within=pyo.NonNegativeReals)  # EUR
        self.opex_annual     = pyo.Var(initialize=1e3, within=pyo.NonNegativeReals)  # EUR/yr

        # ------------------------------------------------------------------
        # 3. Rachford-Rice
        # ------------------------------------------------------------------
        self.rr_con = pyo.Constraint(
            expr=sum(
                self.z[i] * (self.K[i] - 1) / (1 + self.psi * (self.K[i] - 1))
                for i in self.I
            ) == 0
        )

        # ------------------------------------------------------------------
        # 4. CAPEX — Turton (2012), Table A.1, vertical vessel, CS
        #    log10(C_p°) = K1 + K2·log10(V) + K3·(log10(V))²  [USD 2001, V in m³]
        #    CAPEX [EUR] = C_p° × CEPCI_ratio × F_BM × f_indirect × usd_to_eur
        # ------------------------------------------------------------------
        self.eq_capex_purchased = pyo.Constraint(
            expr=self.capex_purchased
            == pyo.exp(
                (
                    _K1_V
                    + _K2_V * pyo.log(self.vessel_volume) / _LN10
                    + _K3_V * (pyo.log(self.vessel_volume) / _LN10) ** 2
                ) * _LN10
            )
        )

        self.eq_capex = pyo.Constraint(
            expr=self.CAPEX
            == self.capex_purchased * _CEPCI_RATIO * _F_BM_VESSEL * f_indirect * usd_to_eur
        )

        # ------------------------------------------------------------------
        # 5. OPEX — maintenance only (Peters, Timmerhaus & West, 5th ed.)
        # ------------------------------------------------------------------
        self.eq_opex = pyo.Constraint(
            expr=self.opex_annual == maintenance * self.CAPEX
        )

    def solve(self, solver_name="ipopt", show_log=False):
        self.dummy_obj = pyo.Objective(expr=1)
        return pyo.SolverFactory(solver_name).solve(self, tee=show_log)

    def display_results(self):
        print(f"\n{'='*55}")
        print(f"{'FLASH SEPARATION SUMMARY':^55}")
        print(f"{'='*55}")
        print(f"Vapor Fraction (ψ): {pyo.value(self.psi):.4f}")
        print(f"Diameter:           {pyo.value(self.diameter):.3f} m")
        print(f"Height:             {pyo.value(self.height):.3f} m")
        print(f"Vessel Volume:      {pyo.value(self.vessel_volume):.3f} m³")
        print(f"Purchased cost:     ${pyo.value(self.capex_purchased):,.0f} USD (2001)")
        print(f"Total CAPEX:        €{pyo.value(self.CAPEX):,.0f}")
        print(f"Annual OPEX (maint):€{pyo.value(self.opex_annual):,.0f}")
        print(f"{'-'*55}")
        V = pyo.value(self.F * self.psi)
        L = pyo.value(self.F * (1 - self.psi))
        print(f"Total Vapor Flow:   {V:.4f} mol/s")
        print(f"Total Liquid Flow:  {L:.4f} mol/s")
        print(f"{'-'*55}")
        print(f"{'Component':<15} | {'x_i (Liquid)':<15} | {'y_i (Vapor)':<15}")
        print(f"{'-'*55}")
        for i in self.I:
            print(f"{i:<15} | {pyo.value(self.x[i]):<15.4f} | {pyo.value(self.y[i]):<15.4f}")
        print(f"{'='*55}\n")
