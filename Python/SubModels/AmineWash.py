# =============================================================================
# MEA Amine Wash — Shortcut Cost Correlation Model
# =============================================================================
#
# Based on:
#   Kim & Léonard (2025), "CO2 capture technologies and shortcut cost
#   correlations for different inlet CO2 concentrations and flow rates.
#   Part 1: Chemical absorption", Int. J. Greenhouse Gas Control 145, 104391.
#
# OVERVIEW
# --------
# Surrogate optimisation model for a 30 wt% MEA post-combustion CO2 capture
# plant.  No compression — captured CO2 is delivered at absorber outlet
# pressure for direct downstream use.
#
# Valid operating range:
#   CO2 inlet concentration:  5 – 50 mol%   (y_CO2 = 0.05 – 0.50 mol frac.)
#   Feed gas flow rate:        4.03 – 1613.81 × 10³ Nm³/h
#   Capture scale:             31 – 1250 kt CO2/y
#
# CAPEX
# -----
#   TEC [M€2023] from regression (Table 14) → CAPEX = 2 × f_installed × TEC × 1e6
#   where f_installed = 3.8 (Lang/Hand K_B → K_I, see AmineWashParameters)
#   and the ×2 is the Guthrie top-level overhead factor (buildings,
#   infrastructure, engineering, working capital, startup — Mitsos VL12,
#   slide 32: K = 2 × K_I), consistent with PSA and membrane submodels.
#
# OPEX [€/h]
# ----------
#   Steam + electricity + cooling water + MEA makeup  (direct utilities)
#   + maintenance_factor × CAPEX / op_hours
#   (same structure as GasPermeationMembraneModel and HeatEXModel)
# =============================================================================

import pyomo.environ as pyo
from pyomo.environ import *
from prettytable import PrettyTable
from configuration import (
    Economics,
    AmineWashParameters,
    AmineWashVariables,
)

_P = AmineWashParameters()
_V = AmineWashVariables()
_E = Economics()

# kW → GJ/h:  1 kW = 3.6e-3 GJ/h
_kW_to_GJ_per_h = 3.6e-3


class AmineWash(pyo.ConcreteModel):
    """
    MEA amine scrubbing CO2 capture model (Pyomo ConcreteModel).

    Interface matches the other MembraneFlowsheet submodels:
        CAPEX  [€]    — total installed capital cost
        OPEX   [€/h]  — operating cost (utilities + maintenance)

    Inputs (fixed or linked by master model)
    -----------------------------------------
    nf_CO2_in   [mol/s]  — CO2 entering the absorber
    nf_inert_in [mol/s]  — inerts (N2, H2O, …) in the flue gas

    Outputs
    -------
    nf_CO2_out      [mol/s]  — CO2 leaving in the scrubbed flue gas
    nf_CO2_captured [mol/s]  — CO2 captured (pure, ready for downstream)
    CAPEX           [€]
    OPEX            [€/h]
    """

    def __init__(
        self,
        capture_rate: float = 0.90,
        location_factor: float = 1.0,
        steam_cost: float | None = None,
        cooling_cost: float = _P.cooling_cost_eur_per_gj,
        boiler_efficiency: float = 0.95,
    ):
        """
        Parameters
        ----------
        capture_rate : float
            Fraction of inlet CO2 captured (default 0.90).
        location_factor : float
            Multiplier on CAPEX for non-US-Gulf locations (default 1.0).
        steam_cost : float or None
            LP steam price [€/GJ].  If None (default), uses the canonical
            value AmineWashParameters.steam_cost_eur_per_gj (23.4 €/GJ,
            electricity-derived), the single source of truth for steam cost
            across the flowsheet.
        cooling_cost : float
            Cooling water price [€/GJ].
        boiler_efficiency : float
            Unused (retained for backward compatibility). The steam price is
            now taken from AmineWashParameters.steam_cost_eur_per_gj.
        """
        super().__init__()
        self._capture_rate    = capture_rate
        self._location_factor = location_factor
        self._cooling_cost    = cooling_cost

        if steam_cost is None:
            # Single source of truth: canonical LP steam price (23.4 €/GJ,
            # electricity-derived) defined in AmineWashParameters.
            self._steam_cost = _P.steam_cost_eur_per_gj
        else:
            self._steam_cost = steam_cost

        self.build_parameters()
        self.build_variables()
        self.build_constraints()

    # -------------------------------------------------------------------------
    def build_parameters(self):
        p = _P

        # TEC correlation
        self.tec_alpha = pyo.Param(initialize=p.tec_alpha)
        self.tec_beta  = pyo.Param(initialize=p.tec_beta)
        self.tec_gamma = pyo.Param(initialize=p.tec_gamma)
        self.tec_n     = pyo.Param(initialize=p.tec_n)
        self.tec_m     = pyo.Param(initialize=p.tec_m)

        # Specific energy correlations
        self.reb_alpha  = pyo.Param(initialize=p.reb_alpha)
        self.reb_beta   = pyo.Param(initialize=p.reb_beta)
        self.reb_n      = pyo.Param(initialize=p.reb_n)
        self.reb_m      = pyo.Param(initialize=p.reb_m)

        self.elec_alpha = pyo.Param(initialize=p.elec_alpha)
        self.elec_beta  = pyo.Param(initialize=p.elec_beta)
        self.elec_n     = pyo.Param(initialize=p.elec_n)
        self.elec_m     = pyo.Param(initialize=p.elec_m)

        self.cool_alpha = pyo.Param(initialize=p.cool_alpha)
        self.cool_beta  = pyo.Param(initialize=p.cool_beta)
        self.cool_n     = pyo.Param(initialize=p.cool_n)
        self.cool_m     = pyo.Param(initialize=p.cool_m)

        # CAPEX / OPEX factors
        self.f_installed       = pyo.Param(initialize=p.f_installed)
        self.location_factor   = pyo.Param(initialize=self._location_factor)
        self.maintenance_factor = pyo.Param(initialize=p.maintenance_factor)

        # Utilities
        self.capture_rate    = pyo.Param(initialize=self._capture_rate)
        self.steam_cost      = pyo.Param(initialize=self._steam_cost)    # €/GJ
        self.cooling_cost    = pyo.Param(initialize=self._cooling_cost)  # €/GJ
        self.price_power     = pyo.Param(initialize=_E.price_power)      # €/kWh
        self.mea_makeup_cost = pyo.Param(initialize=p.mea_makeup_eur_per_tco2)  # €/tCO2
        self.op_hours        = pyo.Param(initialize=_E.operating_hours_per_year)  # h/y

    # -------------------------------------------------------------------------
    def build_variables(self):
        v = _V

        # Inlets
        self.nf_CO2_in   = pyo.Var(bounds=(v.nf_CO2_in_lb,   v.nf_CO2_in_ub),   initialize=v.nf_CO2_in)
        self.nf_inert_in = pyo.Var(bounds=(v.nf_inert_in_lb, v.nf_inert_in_ub), initialize=v.nf_inert_in)

        # Outlets
        self.nf_CO2_out      = pyo.Var(bounds=(v.nf_CO2_out_lb, v.nf_CO2_out_ub), initialize=v.nf_CO2_out)
        self.nf_CO2_captured = pyo.Var(bounds=(v.nf_CO2_cap_lb, v.nf_CO2_cap_ub), initialize=v.nf_CO2_cap)

        # Process intermediates
        self.y_CO2_in    = pyo.Var(bounds=(v.y_CO2_in_lb,    v.y_CO2_in_ub),    initialize=v.y_CO2_in)
        self.F_thousands = pyo.Var(bounds=(v.F_thousands_lb, v.F_thousands_ub),  initialize=v.F_thousands)
        self.scale_tph   = pyo.Var(bounds=(v.scale_tph_lb,   v.scale_tph_ub),    initialize=v.scale_tph)

        # Equipment cost
        self.TEC   = pyo.Var(bounds=(v.TEC_lb,   v.TEC_ub),   initialize=v.TEC)    # M€2023
        self.CAPEX = pyo.Var(bounds=(v.CAPEX_lb, v.CAPEX_ub), initialize=v.CAPEX)  # €

        # Specific energy duties
        self.Q_reboiler_spec = pyo.Var(bounds=(v.Q_reb_spec_lb,  v.Q_reb_spec_ub),  initialize=v.Q_reb_spec)
        self.W_elec_spec     = pyo.Var(bounds=(v.W_elec_spec_lb, v.W_elec_spec_ub), initialize=v.W_elec_spec)
        self.Q_cooling_spec  = pyo.Var(bounds=(v.Q_cool_spec_lb, v.Q_cool_spec_ub), initialize=v.Q_cool_spec)

        # Actual duties [kW]
        self.Q_reboiler_kW = pyo.Var(bounds=(v.Q_reb_kW_lb,  v.Q_reb_kW_ub),  initialize=v.Q_reb_kW)
        self.W_elec_kW     = pyo.Var(bounds=(v.W_elec_kW_lb, v.W_elec_kW_ub), initialize=v.W_elec_kW)
        self.Q_cooling_kW  = pyo.Var(bounds=(v.Q_cool_kW_lb, v.Q_cool_kW_ub), initialize=v.Q_cool_kW)

        # Operating cost [€/h]
        self.OPEX = pyo.Var(bounds=(v.op_cost_lb, v.op_cost_ub), initialize=v.op_cost)

    # -------------------------------------------------------------------------
    def build_constraints(self):

        # ── Mass balances ────────────────────────────────────────────────────
        self.y_CO2_in_c = pyo.Constraint(
            expr=self.y_CO2_in
            == self.nf_CO2_in / (self.nf_CO2_in + self.nf_inert_in)
        )
        self.nf_CO2_captured_c = pyo.Constraint(
            expr=self.nf_CO2_captured == self.capture_rate * self.nf_CO2_in
        )
        self.nf_CO2_out_c = pyo.Constraint(
            expr=self.nf_CO2_out == (1 - self.capture_rate) * self.nf_CO2_in
        )
        # F [10³ Nm³/h] = nf_total [mol/s] × 22.414e-3 [m³/mol] × 3.6 [ks/h]
        self.F_thousands_c = pyo.Constraint(
            expr=self.F_thousands
            == (self.nf_CO2_in + self.nf_inert_in) * 22.414e-3 * 3.6
        )
        # S [tCO2/h] = nf_CO2_captured [mol/s] × 44.01e-3 [kg/mol] × 3.6 [ks/h]
        self.scale_tph_c = pyo.Constraint(
            expr=self.scale_tph == self.nf_CO2_captured * 44.01e-3 * 3.6
        )

        # ── TEC correlation (Kim & Léonard 2025, Table 14) ───────────────────
        self.TEC_c = pyo.Constraint(
            expr=self.TEC
            == self.tec_alpha
            + (self.tec_beta * self.y_CO2_in ** self.tec_n + self.tec_gamma)
            * self.F_thousands ** self.tec_m
        )

        # ── CAPEX [€] = 2 × f_installed × location_factor × TEC [M€] × 1e6 ────
        # 2× is the Guthrie overhead factor (Mitsos VL12 slide 32: K = 2×K_I).
        # f_installed is the Lang/Hand K_B→K_I factor (weighted equipment mix).
        self.CAPEX_c = pyo.Constraint(
            expr=self.CAPEX
            == 2 * self.f_installed * self.location_factor * self.TEC * 1e6
        )

        # ── Specific energy correlations ─────────────────────────────────────
        self.Q_reboiler_spec_c = pyo.Constraint(
            expr=self.Q_reboiler_spec
            == self.reb_alpha * pyo.exp(self.reb_n * self.y_CO2_in)
            + self.reb_beta  * pyo.exp(self.reb_m * self.y_CO2_in)
        )
        self.W_elec_spec_c = pyo.Constraint(
            expr=self.W_elec_spec
            == self.elec_alpha * pyo.exp(self.elec_n * self.y_CO2_in)
            + self.elec_beta   * pyo.exp(self.elec_m * self.y_CO2_in)
        )
        self.Q_cooling_spec_c = pyo.Constraint(
            expr=self.Q_cooling_spec
            == self.cool_alpha * pyo.exp(self.cool_n * self.y_CO2_in)
            + self.cool_beta   * pyo.exp(self.cool_m * self.y_CO2_in)
        )

        # ── Actual duties [kW] ───────────────────────────────────────────────
        # GJ/tCO2 × tCO2/h ÷ 3.6e-3 GJ/(kW·h) = kW
        self.Q_reboiler_kW_c = pyo.Constraint(
            expr=self.Q_reboiler_kW
            == self.Q_reboiler_spec * self.scale_tph / _kW_to_GJ_per_h
        )
        # kWh/tCO2 × tCO2/h = kW
        self.W_elec_kW_c = pyo.Constraint(
            expr=self.W_elec_kW == self.W_elec_spec * self.scale_tph
        )
        self.Q_cooling_kW_c = pyo.Constraint(
            expr=self.Q_cooling_kW
            == self.Q_cooling_spec * self.scale_tph / _kW_to_GJ_per_h
        )

        # ── OPEX [€/h] ───────────────────────────────────────────────────────
        # Utilities:
        #   steam    [€/h] = Q_reb   [kW] × 3.6e-3 [GJ/(kW·h)] × c_steam  [€/GJ]
        #   elec     [€/h] = W_elec  [kW] × c_elec [€/kWh]
        #   cooling  [€/h] = Q_cool  [kW] × 3.6e-3 [GJ/(kW·h)] × c_cw    [€/GJ]
        #   MEA      [€/h] = c_MEA   [€/tCO2] × S  [tCO2/h]
        # Maintenance [€/h] = maint_factor [1/y] × CAPEX [€] / op_hours [h/y]
        self.OPEX_c = pyo.Constraint(
            expr=self.OPEX
            == self.Q_reboiler_kW * _kW_to_GJ_per_h * self.steam_cost
            + self.W_elec_kW * self.price_power
            + self.Q_cooling_kW * _kW_to_GJ_per_h * self.cooling_cost
            + self.mea_makeup_cost * self.scale_tph
            + self.maintenance_factor * self.CAPEX / self.op_hours
        )

    # -------------------------------------------------------------------------
    def print_model(self):
        model = self

        def _v(var) -> float:
            val = pyo.value(var)
            return float(val) if val is not None else 0.0

        print("\nAmine Wash (MEA 30 wt%, 90 % capture)")
        tbl = PrettyTable()
        tbl.field_names = ["Variable", "Value", "Unit"]
        tbl.add_row(["nf_CO2_in",            f"{_v(model.nf_CO2_in):.2f}",         "mol/s"])
        tbl.add_row(["nf_inert_in",           f"{_v(model.nf_inert_in):.2f}",       "mol/s"])
        tbl.add_row(["y_CO2_in",              f"{_v(model.y_CO2_in)*100:.2f}",      "mol%"])
        tbl.add_row(["F",                     f"{_v(model.F_thousands):.2f}",       "10³ Nm³/h"])
        tbl.add_row(["nf_CO2_captured",       f"{_v(model.nf_CO2_captured):.2f}",   "mol/s"])
        tbl.add_row(["nf_CO2_out",            f"{_v(model.nf_CO2_out):.2f}",        "mol/s"])
        tbl.add_row(["capture scale",         f"{_v(model.scale_tph):.2f}",         "tCO2/h"])
        tbl.add_row(["capture scale",         f"{_v(model.scale_tph)*_E.operating_hours_per_year/1e3:.1f}", "kt CO2/y"])
        tbl.add_row(["---", "", ""])
        tbl.add_row(["Q_reboiler (spec)",     f"{_v(model.Q_reboiler_spec):.3f}",   "GJ/tCO2"])
        tbl.add_row(["W_elec (spec)",         f"{_v(model.W_elec_spec):.3f}",       "kWh/tCO2"])
        tbl.add_row(["Q_cooling (spec)",      f"{_v(model.Q_cooling_spec):.3f}",    "GJ/tCO2"])
        tbl.add_row(["Q_reboiler",            f"{_v(model.Q_reboiler_kW)/1e3:.2f}", "MW"])
        tbl.add_row(["W_elec",                f"{_v(model.W_elec_kW)/1e3:.2f}",     "MW"])
        tbl.add_row(["Q_cooling",             f"{_v(model.Q_cooling_kW)/1e3:.2f}",  "MW"])
        tbl.add_row(["---", "", ""])
        tbl.add_row(["TEC",                   f"{_v(model.TEC):.2f}",               "M€"])
        tbl.add_row(["CAPEX",                 f"{_v(model.CAPEX)/1e6:.2f}",         "M€"])
        tbl.add_row(["OPEX",                  f"{_v(model.OPEX):.1f}",              "€/h"])
        tbl.add_row(["OPEX",                  f"{_v(model.OPEX)*_E.operating_hours_per_year/1e6:.2f}", "M€/y"])
        print(tbl)

        # OPEX breakdown
        _scale  = _v(model.scale_tph)
        _steam  = _v(model.Q_reboiler_kW) * _kW_to_GJ_per_h * _v(model.steam_cost)
        _elec   = _v(model.W_elec_kW) * _v(model.price_power)
        _cool   = _v(model.Q_cooling_kW) * _kW_to_GJ_per_h * _v(model.cooling_cost)
        _mea    = _v(model.mea_makeup_cost) * _scale
        _maint  = _v(model.maintenance_factor) * _v(model.CAPEX) / _v(model.op_hours)

        print("\n  OPEX breakdown [€/h]:")
        print(f"    Steam:       {_steam:8.1f}")
        print(f"    Electricity: {_elec:8.1f}")
        print(f"    Cooling:     {_cool:8.1f}")
        print(f"    MEA makeup:  {_mea:8.1f}")
        print(f"    Maintenance: {_maint:8.1f}")
        print(f"    Total:       {_v(model.OPEX):8.1f}")
