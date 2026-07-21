# =============================================================================
# PSA Virtual Moving Bed (VMB) Model — design rationale
# =============================================================================
#
# This model follows the Virtual Moving Bed (VMB) framework of Sees et al. (2021),
# as implemented for CO2/CO separation on zeolite 13X in Tjahjana (2025).
#
# CORE IDEA — VMB LINK
# ---------------------
# In a real PSA unit, a fixed bed alternates between adsorption (at p_H) and
# desorption (at p_L).  The VMB abstraction replaces this with a fictitious
# continuous solid circulation:
#
#   mf_ads  [kg/s]  — equivalent adsorbent circulation rate
#   m_ads   [kg]    — total adsorbent inventory in the unit
#   t_cyc   [s]     — full cycle time  = t_ads + t_des
#
# The link between inventory and circulation rate is:
#
#   mf_ads = m_ads / t_cyc                          ... (VMB link)
#
# Physically: the entire inventory "circulates" once per cycle.
# mf_ads appears in the retentate balances (nf_CO2_HR, nf_CO2_LR, etc.)
# and drives the separation; m_ads sets the equipment cost.
# The two are coupled through t_cyc, so the optimizer sees the cost
# consequence of the solid circulation rate.
#
# PRESSURE-DEPENDENT CYCLE TIMES (improvement over Tjahjana 2025)
# ----------------------------------------------------------------
# Tjahjana fixed t_ads = 48 s regardless of pressure.  Here t_ads and t_des
# are computed from the Park et al. kinetic model (power-law β'(P) fit):
#
#   β'(P)  = exp(β_intercept + β_slope · ln(P))
#   t_ads  = −ln(1 − saturation) · (1 + β'(p_H)) / (ha·ρs·Cs)
#   t_des  = −ln(1 − saturation) · (1 + β'(p_L)) / (ha·ρs·Cs)
#
# This makes t_cyc — and therefore mf_ads, m_ads, and all costs — functions
# of p_H and p_L, allowing the optimizer to exploit the pressure-cycle-time
# trade-off correctly.
#
# WHY NOT A DIRECT WORKING-CAPACITY BALANCE?
# -------------------------------------------
# An alternative formulation sizes m_ads explicitly via:
#
#   m_ads · bed_util · sat · (q_CO2_H − q_CO2_L) = nf_CO2_net · t_cyc
#
# This is mathematically implied by the retentate balances + the VMB link
# (up to the small void-space holdup correction), so adding it as an
# explicit constraint would be redundant and could cause IPOPT to report
# a rank-deficient Jacobian.  It is therefore omitted; the VMB link alone
# is sufficient to size m_ads through the separation duty.
# =============================================================================

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import pi
from prettytable import PrettyTable
from pyomo.environ import units as u
from configuration import (
    Economics,
    Constants,
    PSAParameters,
    PSAVariables,
)


class PSA(pyo.ConcreteModel):
    def __init__(self, allow_vacuum: bool = False, p_product_out_bar: float = None):
        """
        Parameters
        ----------
        allow_vacuum : bool
            If True, the low-pressure column is allowed to operate below 1 bar
            (p_L < 100 kPa).  A vacuum pump is then sized via the Guthrie (1974)
            centrifugal-pump correlation and its power and cost are included in
            OPEX and CAPEX respectively.  If False (default), p_L is constrained
            to ≥ 100 kPa and no pump is added.
        p_product_out_bar : float or None
            If provided, a post-compressor is added that brings the product gas
            from p_H up to this pressure (in bar). The Guthrie (1974) compressor
            correlation is used and its power and cost are added to OPEX/CAPEX.
            Must be greater than p_H; if None, no post-compression is added.
        """
        super().__init__()
        self._allow_vacuum = allow_vacuum
        # Convert bar → kPa and store; None means no post-compression.
        self._p_product_out_kPa = p_product_out_bar * 100.0 if p_product_out_bar is not None else None

        self.build_parameters()
        self.build_variables()
        self.build_constraints()

    def build_parameters(self):
        # PSA Parameters
        self.R = pyo.Param(initialize=Constants.R, units=u.J / u.mol / u.K)
        self.T = pyo.Param(initialize=Constants.T, units=u.K)
        self.p_atm = pyo.Param(initialize=Constants.P / 1000, units=u.kPa)

        self.y_CO_product = pyo.Param(initialize=PSAParameters.y_CO_product)
        self.y_CO2_product = pyo.Param(initialize=PSAParameters.y_CO2_product)
        self.y_CO_psaH_out = pyo.Param(initialize=PSAParameters.y_CO_psaH_out)
        self.y_CO2_psaH_out = pyo.Param(initialize=PSAParameters.y_CO2_psaH_out)

        self.q_CO2_s = pyo.Param(initialize=Constants.q_CO2_s, units=u.mol / u.kg)
        self.Bs_CO2 = pyo.Param(initialize=Constants.Bs_CO2, units=1 / u.kPa)
        self.ns_CO2 = pyo.Param(initialize=Constants.ns_CO2)

        self.q_CO_s = pyo.Param(initialize=Constants.q_CO_s, units=u.mol / u.kg)
        self.Bs_CO = pyo.Param(initialize=Constants.Bs_CO, units=1 / u.kPa)
        self.ns_CO = pyo.Param(initialize=Constants.ns_CO)

        self.Henry_constant_CO2 = pyo.Param(initialize=Constants.Henry_constant_CO2, units=u.mol / u.kg / u.kPa)

        self.bulk_density = pyo.Param(initialize=Constants.bulk_density, units=u.kg / u.m**3)
        self.voidage = pyo.Param(initialize=PSAParameters.voidage)
        self.efficiency_comp = pyo.Param(initialize=PSAParameters.efficiency_comp)
        self.efficiency_pump = pyo.Param(initialize=PSAParameters.efficiency_pump)
        self.park_ha_rhosCs = pyo.Param(initialize=Constants.park_ha_rhosCs)
        self.park_beta_slope = pyo.Param(initialize=Constants.park_beta_slope)
        self.park_beta_intercept = pyo.Param(initialize=Constants.park_beta_intercept)
        self.park_ha_rhosCs_CO = pyo.Param(initialize=Constants.park_ha_rhosCs_CO)
        self.park_beta_slope_CO = pyo.Param(initialize=Constants.park_beta_slope_CO)
        self.park_beta_intercept_CO = pyo.Param(initialize=Constants.park_beta_intercept_CO)
        self.bed_utilisation = pyo.Param(initialize=Constants.bed_utilisation)
        self.price_zeoperkg = pyo.Param(initialize=PSAParameters.price_zeoperkg, units=1 / u.kg)

        # Guthrie parameter for pressure vessel
        self.factor_material = pyo.Param(initialize=PSAParameters.factor_material)
        self.factor_installation = pyo.Param(initialize=PSAParameters.factor_installation)
        self.price_index = pyo.Param(initialize=PSAParameters.price_index)
        self.indirect_cost_factor_pv = pyo.Param(initialize=PSAParameters.indirect_cost_factor_pv)

        # Guthrie parameter for compressor
        self.factor_installation_comp = pyo.Param(initialize=PSAParameters.factor_installation_comp)
        self.indirect_cost_factor_comp = pyo.Param(initialize=PSAParameters.indirect_cost_factor_comp)

        # Turton et al. (2012) liquid ring vacuum pump parameters
        self.turton_K1_pump = pyo.Param(initialize=PSAParameters.turton_K1_pump)
        self.turton_K2_pump = pyo.Param(initialize=PSAParameters.turton_K2_pump)
        self.turton_K3_pump = pyo.Param(initialize=PSAParameters.turton_K3_pump)
        self.turton_FBM_pump = pyo.Param(initialize=PSAParameters.turton_FBM_pump)
        self.indirect_cost_factor_pump = pyo.Param(initialize=PSAParameters.indirect_cost_factor_pump)
        # Heat capacity ratio for isentropic vacuum pump power
        self.gamma_mix = pyo.Param(initialize=PSAParameters.gamma_mix)

        # Fluidization constraint parameters
        # CEPCI scaling factors
        _econ = Economics()
        self.cepci_ratio = pyo.Param(initialize=_econ.CEPCI_2024 / _econ.CEPCI_1974)   # Guthrie (1974) → 2024
        self.cepci_ratio_turton = pyo.Param(initialize=_econ.CEPCI_2024 / _econ.CEPCI_2001)  # Turton (2001) → 2024

        # OPEX maintenance parameters
        self.maint_factor_rotating = pyo.Param(initialize=_econ.maintenance_factor_rotating)  # fraction/yr
        self.maint_factor_static   = pyo.Param(initialize=_econ.maintenance_factor_static)    # fraction/yr
        self.op_hours              = pyo.Param(initialize=_econ.operating_hours_per_year)      # h/yr
        self.adsorbent_lifetime    = pyo.Param(initialize=PSAParameters.adsorbent_lifetime_years)  # yr

    def build_variables(self):
        # Variables
        # saturation_ads: fraction of equilibrium loading reached at end of adsorption (H column).
        # saturation_des: fraction of equilibrium loading remaining at end of desorption (L column),
        #   i.e. how far toward q_CO2(p_L) regeneration proceeds before switching back.
        # Both are free: the optimizer independently trades off cycle time vs adsorbent inventory
        # for each half-step.
        # Species-specific saturation fractions: each species reaches its own equilibrium
        # fraction within the shared adsorption / desorption time (t_ads, t_des).
        # CO is faster than CO2 (higher ha/ρsCs), so for the same t_ads CO reaches a
        # higher saturation — the optimizer chooses t_ads/t_des and all four saturations
        # follow from the Park kinetic constraints.
        self.saturation_ads_CO2 = pyo.Var(bounds=(0.60, 0.9999), initialize=0.75)
        self.saturation_ads_CO  = pyo.Var(bounds=(0.60, 0.9999), initialize=0.85)
        self.saturation_des_CO2 = pyo.Var(bounds=(0.60, 0.9999), initialize=0.75)
        self.saturation_des_CO  = pyo.Var(bounds=(0.60, 0.9999), initialize=0.85)
        # Shared adsorption and desorption step times [s].
        self.t_ads = pyo.Var(bounds=(10.0, 10000.0), initialize=500.0)
        self.t_des = pyo.Var(bounds=(10.0, 10000.0), initialize=250.0)
        self.p_H = pyo.Var(
            bounds=(PSAVariables.p_H_lb, PSAVariables.p_H_ub),
            initialize=PSAVariables.p_H,
        )  # unit: kPa
        # In vacuum mode the low-pressure column may operate below 1 bar.
        # A practical lower limit of 10 kPa (0.1 bar) is used; the Sips isotherm
        # and the log term in pump power remain well-defined above that limit.
        _p_L_lb = 10.0 if self._allow_vacuum else PSAVariables.p_L_lb
        _p_L_ub = PSAVariables.p_L_ub if not self._allow_vacuum else 100.0
        self.p_L = pyo.Var(
            bounds=(_p_L_lb, _p_L_ub),
            initialize=PSAVariables.p_L,
        )  # unit: kPa
        self.nf_CO2_psaH_in = pyo.Var(
            bounds=(PSAVariables.nf_CO2_psaH_in_lb, PSAVariables.nf_CO2_psaH_in_ub),
            initialize=PSAVariables.nf_CO2_psaH_in,
        )  # unit: mol/s
        self.nf_CO_psaH_in = pyo.Var(
            bounds=(PSAVariables.nf_CO_psaH_in_lb, PSAVariables.nf_CO_psaH_in_ub),
            initialize=PSAVariables.nf_CO_psaH_in,
        )  # unit: mol/s
        self.y_CO2_psaH_in = pyo.Var(
            bounds=(PSAVariables.y_CO2_psaH_in_lb, PSAVariables.y_CO2_psaH_in_ub),
            initialize=PSAVariables.y_CO2_psaH_in,
        )  # dimensionless
        self.y_CO_psaH_in = pyo.Var(
            bounds=(PSAVariables.y_CO_psaH_in_lb, PSAVariables.y_CO_psaH_in_ub),
            initialize=PSAVariables.y_CO_psaH_in,
        )  # dimensionless
        self.nf_CO2_psaH_out = pyo.Var(
            bounds=(PSAVariables.nf_CO2_psaH_out_lb, PSAVariables.nf_CO2_psaH_out_ub),
            initialize=PSAVariables.nf_CO2_psaH_out,
        )  # unit: mol/s
        self.nf_CO_psaH_out = pyo.Var(
            bounds=(PSAVariables.nf_CO_psaH_out_lb, PSAVariables.nf_CO_psaH_out_ub),
            initialize=PSAVariables.nf_CO_psaH_out,
        )  # unit: mol/s
        self.nf_CO2_product = pyo.Var(
            bounds=(PSAVariables.nf_CO2_product_lb, PSAVariables.nf_CO2_product_ub),
            initialize=PSAVariables.nf_CO2_product,
        )  # unit: mol/s
        self.nf_CO_product = pyo.Var(
            bounds=(PSAVariables.nf_CO_product_lb, PSAVariables.nf_CO_product_ub),
            initialize=PSAVariables.nf_CO_product,
        )  # unit: mol/s
        self.nf_CO2_psaL_out = pyo.Var(
            bounds=(PSAVariables.nf_CO2_psaL_out_lb, PSAVariables.nf_CO2_psaL_out_ub),
            initialize=PSAVariables.nf_CO2_psaL_out,
        )  # unit: mol/s
        self.nf_CO_psaL_out = pyo.Var(
            bounds=(PSAVariables.nf_CO_psaL_out_lb, PSAVariables.nf_CO_psaL_out_ub),
            initialize=PSAVariables.nf_CO_psaL_out,
        )  # unit: mol/s
        self.y_CO2_psaL_out = pyo.Var(
            bounds=(PSAVariables.y_CO2_psaL_out_lb, PSAVariables.y_CO2_psaL_out_ub),
            initialize=PSAVariables.y_CO2_psaL_out,
        )  # dimensionless
        self.y_CO_psaL_out = pyo.Var(
            bounds=(PSAVariables.y_CO_psaL_out_lb, PSAVariables.y_CO_psaL_out_ub),
            initialize=PSAVariables.y_CO_psaL_out,
        )  # dimensionless
        self.nf_CO2_LR = pyo.Var(
            bounds=(PSAVariables.nf_CO2_LR_lb, PSAVariables.nf_CO2_LR_ub),
            initialize=PSAVariables.nf_CO2_LR,
        )  # unit: mol/s
        self.nf_CO_LR = pyo.Var(
            bounds=(PSAVariables.nf_CO_LR_lb, PSAVariables.nf_CO_LR_ub),
            initialize=PSAVariables.nf_CO_LR,
        )  # unit: mol/s
        self.nf_CO2_HR = pyo.Var(
            bounds=(PSAVariables.nf_CO2_HR_lb, PSAVariables.nf_CO2_HR_ub),
            initialize=PSAVariables.nf_CO2_HR,
        )  # unit: mol/s
        self.nf_CO_HR = pyo.Var(
            bounds=(PSAVariables.nf_CO_HR_lb, PSAVariables.nf_CO_HR_ub),
            initialize=PSAVariables.nf_CO_HR,
        )  # unit: mol/s
        self.mf_ads = pyo.Var(
            bounds=(PSAVariables.mf_ads_lb, PSAVariables.mf_ads_ub),
            initialize=PSAVariables.mf_ads,
        )  # unit: kg/s
        self.q_CO2_H = pyo.Var(
            bounds=(PSAVariables.q_CO2_H_lb, PSAVariables.q_CO2_H_ub),
            initialize=PSAVariables.q_CO2_H,
        )  # unit: mol/kg
        self.q_CO2_L = pyo.Var(
            bounds=(PSAVariables.q_CO2_L_lb, PSAVariables.q_CO2_L_ub),
            initialize=PSAVariables.q_CO2_L,
        )  # unit: mol/kg
        self.q_CO_H = pyo.Var(
            bounds=(PSAVariables.q_CO_H_lb, PSAVariables.q_CO_H_ub),
            initialize=PSAVariables.q_CO_H,
        )  # unit: mol/kg
        self.q_CO_L = pyo.Var(
            bounds=(PSAVariables.q_CO_L_lb, PSAVariables.q_CO_L_ub),
            initialize=PSAVariables.q_CO_L,
        )  # unit: mol/kg
        self.power_comp_in = pyo.Var(
            bounds=(PSAVariables.power_comp_in_lb, PSAVariables.power_comp_in_ub),
            initialize=PSAVariables.power_comp_in,
        )  # unit: kW
        self.power_comp_out = pyo.Var(
            bounds=(PSAVariables.power_comp_out_lb, PSAVariables.power_comp_out_ub),
            initialize=PSAVariables.power_comp_out,
        )  # unit: kW
        self.power_pump_in = pyo.Var(
            bounds=(PSAVariables.power_pump_in_lb, PSAVariables.power_pump_in_ub),
            initialize=PSAVariables.power_comp_in,
        )  # unit: kW
        self.power_pump_out = pyo.Var(
            bounds=(PSAVariables.power_pump_out_lb, PSAVariables.power_pump_out_ub),
            initialize=PSAVariables.power_pump_out,
        )  # unit: kW
        self.op_cost_psa = pyo.Var(
            bounds=(PSAVariables.op_cost_psa_lb, PSAVariables.op_cost_psa_ub),
            initialize=PSAVariables.op_cost_psa,
        )  # unit: €/h
        self.price_adsorbent = pyo.Var(
            bounds=(PSAVariables.price_adsorbent_lb, PSAVariables.price_adsorbent_ub),
            initialize=PSAVariables.price_adsorbent,
        )  # unit: €
        self.m_ads = pyo.Var(
            bounds=(PSAVariables.m_ads_lb, PSAVariables.m_ads_ub),
            initialize=PSAVariables.m_ads,
        )  # unit: kg
        self.V_pv_min = pyo.Var(
            bounds=(PSAVariables.V_pv_min_lb, PSAVariables.V_pv_min_ub),
            initialize=PSAVariables.V_pv_min,
        )  # unit: m^3
        self.Diameter = pyo.Var(
            bounds=(PSAVariables.Diameter_lb, 3.048),
            initialize=PSAVariables.Diameter,
        )  # unit: m  — upper bound = 10 ft, Guthrie (1974) correlation limit
        self.Height = pyo.Var(
            bounds=(PSAVariables.Height_lb, 30.48),
            initialize=PSAVariables.Height,
        )  # unit: m  — upper bound = 100 ft, Guthrie (1974) correlation limit
        self.V_pv = pyo.Var(
            bounds=(PSAVariables.V_pv_lb, PSAVariables.V_pv_ub),
            initialize=PSAVariables.V_pv,
        )  # unit: m^3
        self.factor_pressure = pyo.Var(
            bounds=(PSAVariables.factor_pressure_lb, PSAVariables.factor_pressure_ub),
            initialize=PSAVariables.factor_pressure,
        )  # dimensionless
        self.base_cost_pv = pyo.Var(
            bounds=(PSAVariables.base_cost_pv_lb, PSAVariables.base_cost_pv_ub),
            initialize=PSAVariables.base_cost_pv,
        )  # unit: €
        self.installed_cost_pv = pyo.Var(
            bounds=(PSAVariables.installed_cost_pv_lb, PSAVariables.installed_cost_pv_ub),
            initialize=PSAVariables.installed_cost_pv,
        )  # unit: €
        self.fixed_cost_pv = pyo.Var(
            bounds=(PSAVariables.fixed_cost_pv_lb, PSAVariables.fixed_cost_pv_ub),
            initialize=PSAVariables.fixed_cost_pv,
        )  # unit: €

        # Compressor
        self.base_cost_comp = pyo.Var(
            bounds=(PSAVariables.base_cost_comp_lb, PSAVariables.base_cost_comp_ub),
            initialize=PSAVariables.base_cost_comp,
        )  # unit: €
        self.F_a_cost_comp = pyo.Var(
            bounds=(PSAVariables.F_a_cost_comp_lb, PSAVariables.F_a_cost_comp_ub),
            initialize=PSAVariables.F_a_cost_comp,
        )  # unit: €
        self.installed_cost_comp = pyo.Var(
            bounds=(PSAVariables.installed_cost_comp_lb, PSAVariables.installed_cost_comp_ub),
            initialize=PSAVariables.installed_cost_comp,
        )  # unit: €
        self.fixed_cost_comp = pyo.Var(
            bounds=(PSAVariables.fixed_cost_comp_lb, PSAVariables.fixed_cost_comp_ub),
            initialize=PSAVariables.fixed_cost_comp,
        )  # unit: €

        # Pump

        self.base_cost_pump = pyo.Var(
            bounds=(PSAVariables.base_cost_pump_lb, PSAVariables.base_cost_pump_ub),
            initialize=PSAVariables.base_cost_pump,
        )  # unit: €
        self.installed_cost_pump = pyo.Var(
            bounds=(PSAVariables.installed_cost_pump_lb, PSAVariables.installed_cost_pump_ub),
            initialize=PSAVariables.installed_cost_pump,
        )  # unit: €
        self.fixed_cost_pump = pyo.Var(
            bounds=(PSAVariables.fixed_cost_pump_lb, PSAVariables.fixed_cost_pump_ub),
            initialize=PSAVariables.fixed_cost_pump,
        )  # unit: €

        self.CAPEX = pyo.Var(bounds=(1, None))  # in €
        self.OPEX  = pyo.Var(bounds=(0, None))  # in €/h — electricity + maintenance + adsorbent replacement

        # Number of parallel identical trains (continuous relaxation; round up after solving).
        # Each train has the same D and H; all trains share the same compressor/pump sizing
        # (which is based on total flow). The per-train adsorbent inventory is m_ads / N_trains.
        self.N_trains = pyo.Var(bounds=(1.0, 100.0), initialize=1.0)

        # Low-pressure column pressure vessel cost variables.
        # Same geometry as the H column but with its own pressure factor, because
        # a vacuum vessel must resist external (atmospheric) pressure rather than
        # internal pressure — requiring thicker walls as p_L decreases.
        self.factor_pressure_L   = pyo.Var(bounds=(0.9, 11.75), initialize=1.0)
        self.base_cost_pv_L      = pyo.Var(bounds=(1, 1e10),    initialize=pyo.value(self.base_cost_pv))
        self.installed_cost_pv_L = pyo.Var(bounds=(1, 1e6),     initialize=1.0)
        self.fixed_cost_pv_L     = pyo.Var(bounds=(0, 1e8),     initialize=0.0)

        # Post-compressor (optional): compresses product from p_H to p_product_out.
        if self._p_product_out_kPa is not None:
            self.power_postcomp_out = pyo.Var(bounds=(0.001, 1e6), initialize=1000.0)  # kW isentropic
            self.power_postcomp_in  = pyo.Var(bounds=(0.001, 1e6), initialize=1000.0)  # kW shaft
            self.base_cost_postcomp      = pyo.Var(bounds=(1, 1e9), initialize=1e5)    # USD (1974)
            self.F_a_cost_postcomp       = pyo.Var(bounds=(1, 1e7), initialize=5e5)    # USD (1974)
            self.installed_cost_postcomp = pyo.Var(bounds=(1, 1e8), initialize=1e6)    # USD (1974)
            self.fixed_cost_postcomp     = pyo.Var(bounds=(1, 1e8), initialize=1e6)    # EUR (2024)

    def build_constraints(self):
        # Constraints
        # Mole fractions
        # For the psaH inlet, y_CO2 is defined from the flows; y_CO is expressed as
        # (1 - y_CO2) rather than a second independent ratio. This linearises the y_CO
        # constraint (no division) and reduces coupling in the Jacobian rows.
        self.y_CO2_psaH_in_Const = pyo.Constraint(
            expr=self.y_CO2_psaH_in == self.nf_CO2_psaH_in / (self.nf_CO2_psaH_in + self.nf_CO_psaH_in)
        )
        self.y_CO_psaH_in_Const = pyo.Constraint(
            expr=self.y_CO_psaH_in == 1 - self.y_CO2_psaH_in
        )

        # For the product stream, y_CO2_product and y_CO_product are both fixed Params
        # summing to 1.  Encoding both as equality constraints gives two equations that
        # reduce to the same linear relation in nf_CO2_product and nf_CO_product —
        # the Jacobian is rank-deficient by 1.  Only y_CO2_product_const is kept;
        # y_CO_product_const is dropped because it carries no additional information.
        self.y_CO2_product_const = pyo.Constraint(
            expr=self.y_CO2_product == self.nf_CO2_product / (self.nf_CO2_product + self.nf_CO_product)
        )

        # Same linearisation as psaH inlet: y_CO_psaL_out = 1 - y_CO2_psaL_out
        # avoids a redundant nonlinear ratio constraint.
        self.y_CO2_psaL_out_const = pyo.Constraint(
            expr=self.y_CO2_psaL_out == self.nf_CO2_psaL_out / (self.nf_CO2_psaL_out + self.nf_CO_psaL_out)
        )
        self.y_CO_psaL_out_const = pyo.Constraint(
            expr=self.y_CO_psaL_out == 1 - self.y_CO2_psaL_out
        )
        # Sips-model
        self.q_CO2_H_const = pyo.Constraint(
            expr=self.q_CO2_H
            == self.q_CO2_s
            * (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_H) ** (1 / self.ns_CO2)
            / (
                1
                + (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_H) ** (1 / self.ns_CO2)
                + (self.Bs_CO * self.y_CO_psaH_in * self.p_H) ** (1 / self.ns_CO)
            )
        )
        self.q_CO_H_const = pyo.Constraint(
            expr=self.q_CO_H
            == self.q_CO_s
            * (self.Bs_CO * self.y_CO_psaH_in * self.p_H) ** (1 / self.ns_CO)
            / (
                1
                + (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_H) ** (1 / self.ns_CO2)
                + (self.Bs_CO * self.y_CO_psaH_in * self.p_H) ** (1 / self.ns_CO)
            )
        )
        # Frozen-solid approximation (Sees et al. 2021, p.6): blowdown is fast so
        # the adsorbate composition is "frozen" at the feed composition when pressure
        # drops to p_L. q_CO2_L and q_CO_L are therefore evaluated at the feed
        # composition (y_CO2_psaH_in, y_CO_psaH_in) at p_L, not at the L-column
        # outlet composition — which would be circular and physically incorrect.
        self.q_CO2_L_const = pyo.Constraint(
            expr=self.q_CO2_L
            == self.q_CO2_s
            * (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_L) ** (1 / self.ns_CO2)
            / (
                1
                + (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_L) ** (1 / self.ns_CO2)
                + (self.Bs_CO * self.y_CO_psaH_in * self.p_L) ** (1 / self.ns_CO)
            )
        )
        self.q_CO_L_const = pyo.Constraint(
            expr=self.q_CO_L
            == self.q_CO_s
            * (self.Bs_CO * self.y_CO_psaH_in * self.p_L) ** (1 / self.ns_CO)
            / (
                1
                + (self.Bs_CO2 * self.y_CO2_psaH_in * self.p_L) ** (1 / self.ns_CO2)
                + (self.Bs_CO * self.y_CO_psaH_in * self.p_L) ** (1 / self.ns_CO)
            )
        )
        self.q_const1 = pyo.Constraint(expr=self.q_CO2_H >= self.q_CO_H)
        self.q_const2 = pyo.Constraint(expr=self.q_CO2_L >= self.q_CO_L)
        # Positive working capacity: adsorption at p_H must load more CO2 than desorption at p_L.
        # Without this, the solver can find unphysical solutions where the bed loads more at low
        # pressure than high pressure (negative swing), which violates PSA operating logic.
        self.working_cap_const = pyo.Constraint(expr=self.q_CO2_H >= self.q_CO2_L)

        # Retentat / High-retention mass balances.
        # The adsorbed loading terms are scaled by bed_utilisation to account for the
        # fact that real PSA cycles never reach full equilibrium loading — the adsorption
        # front breaks through before the bed is fully loaded. The void-space (gas-phase)
        # terms are not scaled as they are instantaneous, not cycle-averaged.
        # Frozen-solid approximation: the void-space composition at the end of
        # desorption is also taken at the feed composition (y_CO2_psaH_in), consistent
        # with Sees et al. (2021) Eq. (11) where the full hold-up H_i^L is evaluated
        # at the sweep/reference composition — here the feed under the frozen-solid
        # assumption. Using y_CO2_psaL_out here would be circular (same issue as
        # the isotherm reference fixed above).
        # Residual loading after partial desorption: starting from sat_ads × q_H,
        # the bed moves sat_des fraction of the way toward equilibrium at p_L.
        #   q_residual = (1 - sat_des) × sat_ads × q_H + sat_des × q_L
        # Higher sat_des → more regenerated (lower residual) → larger working capacity.
        # This is physically consistent with t_des = -ln(1-sat_des) × ... being longer
        # for higher sat_des.
        _q_CO2_residual = (1 - self.saturation_des_CO2) * self.saturation_ads_CO2 * self.q_CO2_H \
                        + self.saturation_des_CO2 * self.q_CO2_L
        _q_CO_residual  = (1 - self.saturation_des_CO)  * self.saturation_ads_CO  * self.q_CO_H \
                        + self.saturation_des_CO  * self.q_CO_L
        self.nf_CO2_LR_const = pyo.Constraint(
            expr=self.nf_CO2_LR
            == self.mf_ads
            * (
                self.bed_utilisation * _q_CO2_residual
                + self.voidage * self.y_CO2_psaH_in * self.p_L * 1000 / (self.bulk_density * self.R * self.T)
            )
        )
        self.nf_CO_LR_const = pyo.Constraint(
            expr=self.nf_CO_LR
            == self.mf_ads
            * (
                self.bed_utilisation * _q_CO_residual
                + self.voidage * self.y_CO_psaH_in * self.p_L * 1000 / (self.bulk_density * self.R * self.T)
            )
        )
        self.nf_CO2_HR_const = pyo.Constraint(
            expr=self.nf_CO2_HR
            == self.mf_ads
            * (
                self.bed_utilisation * self.q_CO2_H * self.saturation_ads_CO2
                + self.voidage * self.y_CO2_psaH_in * self.p_H * 1000 / (self.bulk_density * self.R * self.T)
            )
        )
        self.nf_CO_HR_const = pyo.Constraint(
            expr=self.nf_CO_HR
            == self.mf_ads
            * (
                self.bed_utilisation * self.q_CO_H * self.saturation_ads_CO
                + self.voidage * self.y_CO_psaH_in * self.p_H * 1000 / (self.bulk_density * self.R * self.T)
            )
        )

        # Balance
        self.nf_CO2_psaH_out_const = pyo.Constraint(
            expr=self.nf_CO2_psaH_out == self.nf_CO2_psaH_in - (self.nf_CO2_HR - self.nf_CO2_LR)
        )
        self.nf_CO_psaH_out_const = pyo.Constraint(
            expr=self.nf_CO_psaH_out == self.nf_CO_psaH_in - (self.nf_CO_HR - self.nf_CO_LR)
        )
        self.nf_CO2_product_const = pyo.Constraint(expr=self.nf_CO2_product == self.nf_CO2_psaH_out)
        self.nf_CO_product_const = pyo.Constraint(expr=self.nf_CO_product == self.nf_CO_psaH_out)
        self.nf_CO2_psaL_out_const = pyo.Constraint(expr=self.nf_CO2_psaL_out == self.nf_CO2_HR - self.nf_CO2_LR)
        self.nf_CO_psaL_out_const = pyo.Constraint(expr=self.nf_CO_psaL_out == self.nf_CO_HR - self.nf_CO_LR)

        # Power
        self.power_comp_out_const = pyo.Constraint(
            expr=self.power_comp_out
            == (self.nf_CO2_psaH_in + self.nf_CO_psaH_in) * self.R * self.T * pyo.log(self.p_H / 100) / 1000
        )
        self.power_comp_in_const = pyo.Constraint(expr=self.power_comp_in == self.power_comp_out / self.efficiency_comp)

        if not self._allow_vacuum:
            # Desorption at atmospheric pressure — no vacuum pump needed.
            # Enforce p_L ≥ p_atm and fix all pump variables to zero.
            self.p_L_atm_const = pyo.Constraint(expr=self.p_L >= 100.0)
            self.power_pump_out.fix(0.0)
            self.power_pump_in.fix(0.0)
            self.op_cost_psa_const = pyo.Constraint(
                expr=self.op_cost_psa == self.power_comp_in * Economics.price_power
            )  # in €/h
        else:
            # Vacuum mode: p_L < 100 kPa.  A liquid ring vacuum pump compresses the
            # desorption gas from p_L back to atmospheric pressure (100 kPa).
            # Power is isentropic ideal-gas work with γ_mix for CO2/CO mixture:
            #   W_is = γ/(γ-1) * n*R*T * [(p_out/p_in)^((γ-1)/γ) - 1]
            # This replaces the previous isothermal formula n*R*T*ln(p_out/p_in),
            # which underestimates power by ~15-20% at compression ratios of 2.
            _gamma = self.gamma_mix
            _nf_pump = self.nf_CO2_psaL_out + self.nf_CO_psaL_out
            self.power_pump_out_const = pyo.Constraint(
                expr=self.power_pump_out
                == (_gamma / (_gamma - 1))
                * _nf_pump * self.R * self.T
                * ((100.0 / self.p_L) ** ((_gamma - 1) / _gamma) - 1)
                / 1000
            )  # kW isentropic
            self.power_pump_in_const = pyo.Constraint(
                expr=self.power_pump_in == self.power_pump_out / self.efficiency_pump
            )
            self.op_cost_psa_const = pyo.Constraint(
                expr=self.op_cost_psa
                == (self.power_comp_in + self.power_pump_in) * Economics.price_power
            )  # in €/h

        # Capital cost adsorbent
        # Park et al. Eq. 16: both species share the same t_ads and t_des (one cycle time).
        # Each species' saturation fraction is determined by how far it approaches
        # equilibrium within that shared time.  CO is ~2× faster than CO2 (higher ha/ρsCs),
        # so for the same t_ads CO reaches a higher saturation — the optimizer is free to
        # choose t_ads and t_des as continuous variables; all four saturations then follow
        # from the four equality constraints below.
        _beta_ads_CO2 = pyo.exp(self.park_beta_intercept    + self.park_beta_slope    * pyo.log(self.p_H))
        _beta_des_CO2 = pyo.exp(self.park_beta_intercept    + self.park_beta_slope    * pyo.log(self.p_L))
        _beta_ads_CO  = pyo.exp(self.park_beta_intercept_CO + self.park_beta_slope_CO * pyo.log(self.p_H))
        _beta_des_CO  = pyo.exp(self.park_beta_intercept_CO + self.park_beta_slope_CO * pyo.log(self.p_L))
        self.t_ads_CO2_const = pyo.Constraint(
            expr=self.t_ads == -pyo.log(1 - self.saturation_ads_CO2) * (1 + _beta_ads_CO2) / self.park_ha_rhosCs
        )
        self.t_ads_CO_const = pyo.Constraint(
            expr=self.t_ads == -pyo.log(1 - self.saturation_ads_CO) * (1 + _beta_ads_CO) / self.park_ha_rhosCs_CO
        )
        self.t_des_CO2_const = pyo.Constraint(
            expr=self.t_des == -pyo.log(1 - self.saturation_des_CO2) * (1 + _beta_des_CO2) / self.park_ha_rhosCs
        )
        self.t_des_CO_const = pyo.Constraint(
            expr=self.t_des == -pyo.log(1 - self.saturation_des_CO) * (1 + _beta_des_CO) / self.park_ha_rhosCs_CO
        )
        # Direct fixed-bed mass balance: CO2 captured per cycle = CO2 fed per cycle.
        #
        #   m_ads [kg] × bed_utilisation × saturation × (q_CO2_H − q_CO2_L) [mol/kg]
        #       = nf_CO2_net [mol/s] × (t_ads + t_des + t_overhead) [s]
        #
        # This replaces the previous mf_ads-based formula which allowed the optimizer
        # to choose low saturation_adsorbent (short cycles, small m_ads) and compensate
        # with large mf_ads (fast solid circulation), which carries no cost.  The direct
        # formula makes m_ads blow up as the working capacity (q_CO2_H − q_CO2_L) → 0,
        # correctly penalising low-pressure-ratio operation.
        #
        # VMB link: mf_ads (circulation rate in retentate balances) = inventory / cycle time.
        # Tjahjana had this with fixed t_ads; here t_ads and t_des are pressure-dependent
        # (Park kinetics), so m_ads inherits the correct pressure dependence via mf_ads.
        self.mf_ads_link_const = pyo.Constraint(
            expr=self.mf_ads == self.m_ads / (self.t_ads + self.t_des)
        )
        self.price_adsorbent_const = pyo.Constraint(expr=self.price_adsorbent == self.m_ads * self.price_zeoperkg)

        # Capital cost pressure vessel
        # Per-train adsorbent volume: total inventory split equally across N_trains identical trains.
        self.V_pv_min_const = pyo.Constraint(expr=self.V_pv_min == self.m_ads / (self.bulk_density * self.N_trains))
        self.V_pv_const1 = pyo.Constraint(expr=self.V_pv == pi * (self.Diameter**2) * self.Height / 4)
        self.V_pv_const2 = pyo.Constraint(expr=self.V_pv >= self.V_pv_min)
        # Minimum aspect ratio: PSA columns must be column-style (tall relative to diameter)
        # to achieve a sharp adsorption front and avoid premature breakthrough.
        # H/D >= 5 is a conservative lower bound for industrial PSA design.
        self.aspect_ratio_const = pyo.Constraint(expr=self.Height >= 5 * self.Diameter)
        # Guthrie (1974) pressure factor polynomial.
        # Both H and L vessels are designed to the same worst-case pressure:
        #   p_H          — internal pressure in the adsorption column
        #   100 / p_L    — equivalent external pressure surrogate for the vacuum column
        #                  (deeper vacuum → thicker walls, analogous to a high-pressure vessel)
        # A smooth-max selects whichever is more demanding:
        #   smooth_max(a, b) = (a + b + sqrt((a-b)² + ε)) / 2
        # A second smooth-max clamps the resulting Guthrie polynomial to Fp >= 1
        # (the Guthrie base-case for a low-pressure carbon-steel vessel).
        _p_eff_H = self.p_H
        _p_eff_L = 100.0 / self.p_L   # kPa surrogate: deeper vacuum → larger value
        _p_eff   = (_p_eff_H + _p_eff_L + pyo.sqrt((_p_eff_H - _p_eff_L)**2 + 1e-2)) / 2
        _poly = 2e-13 * _p_eff**3 - 1e-08 * _p_eff**2 + 0.0006 * _p_eff + 0.9081
        self.factor_pressure_const = pyo.Constraint(
            expr=self.factor_pressure == (_poly + 1 + pyo.sqrt((_poly - 1)**2 + 1e-2)) / 2
        )
        self.base_cost_pv_const = pyo.Constraint(
            # Guthrie (1974) correlation — coefficient is 10^3 (kUSD), NOT 10^6.
            # The original bachelor thesis (Tjahjana, 2025) erroneously used 10^6,
            # inflating vessel CAPEX by a factor of 1000.
            expr=self.base_cost_pv == 10**3 * (self.Height / 1.2192) ** 0.81 * (self.Diameter / 0.9144) ** 1.05
        )  # USD (1974)
        self.installed_cost_pv_const = pyo.Constraint(
            expr=self.installed_cost_pv
            == self.factor_installation
            * self.factor_material
            * self.factor_pressure
            * self.price_index
            * self.base_cost_pv
        )  # USD (1974)
        # Both H and L columns share the same geometry and worst-case pressure factor.
        # Factor of 2 accounts for one H vessel + one L vessel per train; N_trains is
        # applied in the CAPEX constraint.
        self.fixed_cost_pv_const = pyo.Constraint(
            expr=self.fixed_cost_pv
            == Economics.convert_usd_to_euro * self.indirect_cost_factor_pv * self.installed_cost_pv
            * 2 * self.cepci_ratio
        )  # Euro (2 vessels per train, before N_trains scaling)

        # L-column cost variables are kept for backwards compatibility but are now
        # identical to the H-column (same geometry, same worst-case pressure factor).
        self.factor_pressure_L_const = pyo.Constraint(
            expr=self.factor_pressure_L == self.factor_pressure
        )
        self.base_cost_pv_L_const = pyo.Constraint(
            expr=self.base_cost_pv_L == self.base_cost_pv
        )
        self.installed_cost_pv_L_const = pyo.Constraint(
            expr=self.installed_cost_pv_L == self.installed_cost_pv
        )
        self.fixed_cost_pv_L_const = pyo.Constraint(
            expr=self.fixed_cost_pv_L == 0.0
        )  # cost already counted in fixed_cost_pv (factor of 2 above)

        # Capital cost compressor
        self.base_cost_comp_const = pyo.Constraint(
            expr=self.base_cost_comp == 23000 * (self.power_comp_out / 73.55) ** 0.77
        )  # I think in USD
        self.F_a_cost_comp_const = pyo.Constraint(
            expr=self.F_a_cost_comp == 776.5 * (self.base_cost_comp + 1e-6) ** 0.493
        )  # I think in USD # small epsilon for numerical reasons
        self.installed_cost_comp_const = pyo.Constraint(
            expr=self.installed_cost_comp
            == self.factor_installation_comp * self.price_index * (self.base_cost_comp + self.F_a_cost_comp)
        )  # I think in USD
        self.fixed_cost_comp_const = pyo.Constraint(
            expr=self.fixed_cost_comp
            == Economics.convert_usd_to_euro * self.indirect_cost_factor_comp * self.installed_cost_comp
            * self.cepci_ratio
        )  # Euro

        if not self._allow_vacuum:
            # No pump — fix all pump cost variables to zero.
            self.base_cost_pump.fix(0.0)
            self.installed_cost_pump.fix(0.0)
            self.fixed_cost_pump.fix(0.0)
        else:
            # Turton et al. (2012) liquid ring vacuum pump correlation.
            # log10(Cp_USD_2001) = K1 + K2*log10(W_kW) + K3*(log10(W_kW))^2
            # Expressed in natural-log form for IPOPT:
            #   ln(Cp) = ln(10) * [K1 + K2*(ln(W)/ln(10)) + K3*(ln(W)/ln(10))^2]
            _ln10 = pyo.log(10.0)
            _log10_W = pyo.log(self.power_pump_in) / _ln10  # shaft power in kW
            self.base_cost_pump_const = pyo.Constraint(
                expr=pyo.log(self.base_cost_pump)
                == _ln10 * (
                    self.turton_K1_pump
                    + self.turton_K2_pump * _log10_W
                    + self.turton_K3_pump * _log10_W**2
                )
            )  # ln(USD 2001)
            self.installed_cost_pump_const = pyo.Constraint(
                expr=self.installed_cost_pump == self.turton_FBM_pump * self.base_cost_pump
            )  # USD (2001), bare module
            self.fixed_cost_pump_const = pyo.Constraint(
                expr=self.fixed_cost_pump
                == Economics.convert_usd_to_euro
                * self.indirect_cost_factor_pump
                * self.installed_cost_pump
                * self.cepci_ratio_turton
            )  # Euro (2024)

        # Post-compressor (optional): product gas compressed from p_H to p_product_out.
        if self._p_product_out_kPa is not None:
            _p_out = self._p_product_out_kPa  # kPa, fixed user value
            # Isentropic power to compress total product molar flow from p_H to p_out.
            self.power_postcomp_out_const = pyo.Constraint(
                expr=self.power_postcomp_out
                == (self.nf_CO2_product + self.nf_CO_product)
                * self.R * self.T * pyo.log(_p_out / self.p_H) / 1000
            )  # kW
            self.power_postcomp_in_const = pyo.Constraint(
                expr=self.power_postcomp_in == self.power_postcomp_out / self.efficiency_comp
            )
            # Guthrie (1974) compressor correlation, same as main compressor.
            self.base_cost_postcomp_const = pyo.Constraint(
                expr=self.base_cost_postcomp == 23000 * (self.power_postcomp_out / 73.55) ** 0.77
            )  # USD (1974)
            self.F_a_cost_postcomp_const = pyo.Constraint(
                expr=self.F_a_cost_postcomp == 776.5 * (self.base_cost_postcomp + 1e-6) ** 0.493
            )
            self.installed_cost_postcomp_const = pyo.Constraint(
                expr=self.installed_cost_postcomp
                == self.factor_installation_comp * self.price_index
                * (self.base_cost_postcomp + self.F_a_cost_postcomp)
            )  # USD (1974)
            self.fixed_cost_postcomp_const = pyo.Constraint(
                expr=self.fixed_cost_postcomp
                == Economics.convert_usd_to_euro * self.indirect_cost_factor_comp
                * self.installed_cost_postcomp * self.cepci_ratio
            )  # EUR (2024)
            # Update OPEX to include post-compressor electricity.
            # op_cost_psa_const was already set above — deactivate and replace it.
            self.op_cost_psa_const.deactivate()
            if not self._allow_vacuum:
                self.op_cost_psa_const2 = pyo.Constraint(
                    expr=self.op_cost_psa
                    == (self.power_comp_in + self.power_postcomp_in) * Economics.price_power
                )
            else:
                self.op_cost_psa_const2 = pyo.Constraint(
                    expr=self.op_cost_psa
                    == (self.power_comp_in + self.power_pump_in + self.power_postcomp_in) * Economics.price_power
                )
            _postcomp_capex = self.fixed_cost_postcomp
        else:
            _postcomp_capex = 0.0

        # CO recovery floor: at least 60% of the CO fed must leave in the product stream.
        # self.co_recovery_const = pyo.Constraint(
        #     expr=self.nf_CO_product >= 0.60 * self.nf_CO_psaH_in
        # )

        _pump_capex = self.fixed_cost_pump if self._allow_vacuum else 0.0
        # Vessel cost scales linearly with number of trains; compressor/pump/post-comp
        # are sized for the total flow and are not multiplied (one shared unit).
        self.CAPEX_const = pyo.Constraint(
            expr=self.CAPEX
            == self.N_trains * (self.fixed_cost_pv + self.fixed_cost_pv_L) + self.price_adsorbent
            + self.fixed_cost_comp + _pump_capex + _postcomp_capex
        )

        _capex_rotating = self.fixed_cost_comp + _pump_capex + _postcomp_capex
        _capex_static   = self.N_trains * (self.fixed_cost_pv + self.fixed_cost_pv_L)
        self.OPEX_const = pyo.Constraint(
            expr=self.OPEX
            == self.op_cost_psa
            + self.maint_factor_rotating * _capex_rotating / self.op_hours
            + self.maint_factor_static   * _capex_static   / self.op_hours
            + self.price_adsorbent / (self.adsorbent_lifetime * self.op_hours)
        )

    def print_model(self):
        model = self

        print("PSA")
        table_psa = PrettyTable()
        table_psa.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
        table_psa.add_row(["N_trains", model.N_trains.value, "-"])
        table_psa.add_row(["mf_ads", model.mf_ads.value, "kg/s"])
        table_psa.add_row(["m_ads", model.m_ads.value, "kg"])
        table_psa.add_row(["m_ads per train", model.m_ads.value / model.N_trains.value, "kg"])
        table_psa.add_row(["p_H", model.p_H.value / 100, "bar"])
        table_psa.add_row(["p_L", model.p_L.value / 100, "bar"])
        table_psa.add_row(["nf_CO2_psaH_in", model.nf_CO2_psaH_in.value, "mol/s"])
        table_psa.add_row(["nf_CO_psaH_in", model.nf_CO_psaH_in.value, "mol/s"])
        table_psa.add_row(["y_CO2_psaH_in", model.y_CO2_psaH_in.value, "-"])
        table_psa.add_row(["y_CO_psaH_in", model.y_CO_psaH_in.value, "-"])
        table_psa.add_row(["nf_CO2_psaH_out", model.nf_CO2_psaH_out.value, "mol/s"])
        table_psa.add_row(["nf_CO_psaH_out", model.nf_CO_psaH_out.value, "mol/s"])
        table_psa.add_row(["y_CO2_psaH_out", model.y_CO2_psaH_out.value, "-"])
        table_psa.add_row(["y_CO_psaH_out", model.y_CO_psaH_out.value, "-"])
        table_psa.add_row(["nf_CO2_psaL_out", model.nf_CO2_psaL_out.value, "mol/s"])
        table_psa.add_row(["nf_CO_psaL_out", model.nf_CO_psaL_out.value, "mol/s"])
        table_psa.add_row(["q_CO2_H", model.q_CO2_H.value, "mol/kg"])
        table_psa.add_row(["q_CO_H", model.q_CO_H.value, "mol/kg"])
        table_psa.add_row(["q_CO2_L", model.q_CO2_L.value, "mol/kg"])
        table_psa.add_row(["q_CO_L", model.q_CO_L.value, "mol/kg"])
        table_psa.add_row(["nf_CO2_HR", model.nf_CO2_HR.value, "mol/s"])
        table_psa.add_row(["nf_CO_HR", model.nf_CO_HR.value, "mol/s"])
        table_psa.add_row(["nf_CO2_LR", model.nf_CO2_LR.value, "mol/s"])
        table_psa.add_row(["nf_CO_LR", model.nf_CO_LR.value, "mol/s"])
        table_psa.add_row(["nf_CO2_product", model.nf_CO2_product.value, "mol/s"])
        table_psa.add_row(["nf_CO_product", model.nf_CO_product.value, "mol/s"])
        table_psa.add_row(["power_comp_out", model.power_comp_out.value, "kW"])
        table_psa.add_row(["power_comp_in", model.power_comp_in.value, "kW"])
        if self._allow_vacuum:
            table_psa.add_row(["power_pump_out", model.power_pump_out.value, "kW"])
            table_psa.add_row(["power_pump_in", model.power_pump_in.value, "kW"])
        if self._p_product_out_kPa is not None:
            table_psa.add_row(["p_product_out", self._p_product_out_kPa / 100, "bar"])
            table_psa.add_row(["power_postcomp_out", model.power_postcomp_out.value, "kW"])
            table_psa.add_row(["power_postcomp_in", model.power_postcomp_in.value, "kW"])
        table_psa.add_row(["op_cost_psa", model.op_cost_psa.value, "€/h"])
        table_psa.add_row(["Diameter", model.Diameter.value, "m"])
        table_psa.add_row(["Height", model.Height.value, "m"])
        table_psa.add_row(["Fp (H col)", model.factor_pressure.value, "-"])
        table_psa.add_row(["Fp (L col)", model.factor_pressure_L.value, "-"])

        # Adsorption / desorption times and per-species saturations
        table_psa.add_row(["saturation_ads_CO2", pyo.value(model.saturation_ads_CO2), "-"])
        table_psa.add_row(["saturation_ads_CO",  pyo.value(model.saturation_ads_CO),  "-"])
        table_psa.add_row(["saturation_des_CO2", pyo.value(model.saturation_des_CO2), "-"])
        table_psa.add_row(["saturation_des_CO",  pyo.value(model.saturation_des_CO),  "-"])
        table_psa.add_row(["t_ads", pyo.value(model.t_ads), "s"])
        table_psa.add_row(["t_des", pyo.value(model.t_des), "s"])
        table_psa.add_row(["t_cyc", pyo.value(model.t_ads) + pyo.value(model.t_des), "s"])

        # Adsorbent loading utilisation: η = q / q_sat (Sips saturation capacity)
        q_CO2_s = pyo.value(model.q_CO2_s)
        q_CO_s  = pyo.value(model.q_CO_s)
        eta_CO2_H = pyo.value(model.q_CO2_H) / q_CO2_s
        eta_CO2_L = pyo.value(model.q_CO2_L) / q_CO2_s
        eta_CO_H  = pyo.value(model.q_CO_H)  / q_CO_s
        eta_CO_L  = pyo.value(model.q_CO_L)  / q_CO_s
        table_psa.add_row(["eta_CO2 (H col)", eta_CO2_H, "-"])
        table_psa.add_row(["eta_CO2 (L col)", eta_CO2_L, "-"])
        table_psa.add_row(["eta_CO  (H col)", eta_CO_H,  "-"])
        table_psa.add_row(["eta_CO  (L col)", eta_CO_L,  "-"])
        print(table_psa)

        # ------------------------------------------------------------------
        # Economics summary
        # ------------------------------------------------------------------
        _econ = Economics()
        op_hours = _econ.operating_hours_per_year

        def _val(var): return float(pyo.value(var) or 0.0)

        _capex_pv       = _val(model.fixed_cost_pv)
        _capex_pv_L     = _val(model.fixed_cost_pv_L)
        _capex_comp     = _val(model.fixed_cost_comp)
        _capex_pump     = _val(model.fixed_cost_pump)
        _capex_postcomp = _val(model.fixed_cost_postcomp) if self._p_product_out_kPa is not None else 0.0
        _capex_ads      = _val(model.price_adsorbent)
        _capex_tot      = _val(model.CAPEX)

        _opex_elec_yr   = _val(model.op_cost_psa) * op_hours
        _capex_rotating = _capex_comp + _capex_pump + _capex_postcomp
        _capex_static   = _val(model.N_trains) * (_capex_pv + _capex_pv_L)
        _maint_rot      = _econ.maintenance_factor_rotating * _capex_rotating
        _maint_stat     = _econ.maintenance_factor_static   * _capex_static
        _ads_replace_yr = _capex_ads / PSAParameters.adsorbent_lifetime_years
        _opex_tot_yr    = _val(model.OPEX) * op_hours

        print("\nPSA Economics")
        tbl_econ = PrettyTable()
        tbl_econ.field_names = ["Item", "Value [€]", "Share [%]"]
        tbl_econ.align["Item"] = "l"
        tbl_econ.align["Value [€]"] = "r"
        tbl_econ.align["Share [%]"] = "r"

        def _pct(x, total):
            return f"{100 * x / total:.1f}" if total else "—"

        # Vessel cost breakdown
        _D          = pyo.value(model.Diameter)
        _H          = pyo.value(model.Height)
        _N          = pyo.value(model.N_trains)
        _base_pv    = pyo.value(model.base_cost_pv)
        _inst_pv    = pyo.value(model.installed_cost_pv)
        _f_inst     = pyo.value(model.factor_installation)
        _f_mat      = pyo.value(model.factor_material)
        _f_pres     = pyo.value(model.factor_pressure)
        _f_ind      = pyo.value(model.indirect_cost_factor_pv)
        _cepci      = pyo.value(model.cepci_ratio)
        _usd_eur    = Economics.convert_usd_to_euro
        _cost_per_vessel = _f_ind * _inst_pv * _cepci * _usd_eur
        _cost_per_train  = 2 * _cost_per_vessel   # H + L column

        print("\n  --- Pressure Vessel Cost Breakdown ---")
        print(f"  Diameter                          = {_D:.3f} m")
        print(f"  Height                            = {_H:.3f} m")
        print(f"  N_trains                          = {_N:.2f}")
        print(f"  Base cost (Guthrie 1974)          = ${_base_pv:>12,.0f}  USD (1974)")
        print(f"  Factor installation (f_install)   = {_f_inst:.2f}")
        print(f"  Factor material     (f_material)  = {_f_mat:.2f}")
        print(f"  Factor pressure     (f_pressure)  = {_f_pres:.4f}")
        print(f"  Installed cost                    = ${_inst_pv:>12,.0f}  USD (1974)")
        print(f"  Indirect cost factor              = {_f_ind:.2f}")
        print(f"  CEPCI ratio (2024/1974)           = {_cepci:.2f}")
        print(f"  USD -> EUR conversion             = {_usd_eur:.4f}")
        print(f"  Cost per vessel (EUR 2024)        = €{_cost_per_vessel:>12,.0f}")
        print(f"  Cost per train (H+L, EUR 2024)    = €{_cost_per_train:>12,.0f}")
        print(f"  Total (all trains, EUR 2024)      = €{_capex_pv:>12,.0f}")

        tbl_econ.add_row(["--- CAPEX ---", "", ""])
        tbl_econ.add_row(["  Pressure vessels (H+L, all trains)", f"{_capex_pv:,.0f}", _pct(_capex_pv, _capex_tot)])
        tbl_econ.add_row(["  Compressor",               f"{_capex_comp:,.0f}", _pct(_capex_comp, _capex_tot)])
        if self._allow_vacuum:
            tbl_econ.add_row(["  Vacuum pump",          f"{_capex_pump:,.0f}", _pct(_capex_pump, _capex_tot)])
        if self._p_product_out_kPa is not None:
            tbl_econ.add_row(["  Post-compressor",      f"{_capex_postcomp:,.0f}", _pct(_capex_postcomp, _capex_tot)])
        tbl_econ.add_row(["  Adsorbent",         f"{_capex_ads:,.0f}",  _pct(_capex_ads,  _capex_tot)])
        tbl_econ.add_row(["  TOTAL CAPEX",       f"{_capex_tot:,.0f}",  "100.0"])
        tbl_econ.add_row(["--- OPEX (annual) ---", "", ""])
        if self._p_product_out_kPa is not None:
            _elec_label = "  Electricity (comp+pump+postcomp)" if self._allow_vacuum else "  Electricity (comp+postcomp)"
        else:
            _elec_label = "  Electricity (comp+pump)" if self._allow_vacuum else "  Electricity (comp)"
        tbl_econ.add_row([_elec_label,            f"{_opex_elec_yr:,.0f}", _pct(_opex_elec_yr, _opex_tot_yr)])
        tbl_econ.add_row(["  Maintenance (rot)",      f"{_maint_rot:,.0f}",      _pct(_maint_rot,      _opex_tot_yr)])
        tbl_econ.add_row(["  Maintenance (ves)",      f"{_maint_stat:,.0f}",     _pct(_maint_stat,     _opex_tot_yr)])
        tbl_econ.add_row(["  Adsorbent replacement",  f"{_ads_replace_yr:,.0f}", _pct(_ads_replace_yr, _opex_tot_yr)])
        tbl_econ.add_row(["  TOTAL OPEX/yr",          f"{_opex_tot_yr:,.0f}",    "100.0"])
        tbl_econ.add_row(["--- Lifecycle (20 yr) ---", "", ""])
        tbl_econ.add_row(["  Lifecycle cost",    f"{_capex_tot + 20 * _opex_tot_yr:,.0f}", ""])
        print(tbl_econ)
