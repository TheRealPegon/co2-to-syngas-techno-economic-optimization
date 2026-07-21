import pyomo.environ as pyo
from pyomo.environ import units as u, asinh, log, exp, sqrt
from configuration import (
    Economics,
    ElectrolyzerParametersSurrogate,
    ElectrolyzerVariablesSurrogate,
    Constants,
    ElectrolyzerParameters,
)
from prettytable import PrettyTable


class CO2Electrolyzer(pyo.ConcreteModel):
    """Creates the CO2Electrolyzer submodel based on the values in dataclasses in configuration.py"""

    def __init__(self):
        super().__init__()

        self.build_parameters()
        self.build_variables()
        self.build_constraints()

    def build_parameters(self):
        # Electrolyser Parameters
        self.E_anode = pyo.Param(initialize=ElectrolyzerParametersSurrogate.E_anode)  # unit: V

        self.BV_ECO = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_ECO)  # unit: V
        self.BV_CO2ref = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_CO2ref)
        self.BV_CO2m = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_mCO2)
        self.BV_iCO = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_iCO)
        self.BV_iH2 = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_iH2)
        self.BV_CO = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_CO)
        self.BV_H2 = pyo.Param(initialize=ElectrolyzerParametersSurrogate.BV_H2)

        self.Channel_H = pyo.Param(initialize=ElectrolyzerParametersSurrogate.Channel_H)  # unit: m
        self.Channel_W = pyo.Param(initialize=ElectrolyzerParametersSurrogate.Channel_W)  # unit: m
        self.Channel_L = pyo.Param(initialize=ElectrolyzerParametersSurrogate.Channel_L)  # unit: m
        self.Lm = pyo.Param(initialize=ElectrolyzerParametersSurrogate.Lm)  # unit: m
        self.sigma_el = pyo.Param(initialize=ElectrolyzerParametersSurrogate.sigma_el)  # unit: S/m
        self.sigma_m = pyo.Param(initialize=ElectrolyzerParametersSurrogate.sigma_m)  # unit: S/m

        self.reacting_area = pyo.Param(initialize=ElectrolyzerParametersSurrogate.reacting_area)  # unit: m2

        # Electrolyzer stack manufacturing cost [€/m²].
        # Source: Otto & Steenbuck (2025), Table 4.2 at 500 000 m²/a.
        self.base_cost_elec = pyo.Param(initialize=ElectrolyzerParameters.base_cost_elec)  # €/m²

        # Installation factor: stack K_B → installed system K_I.
        # f_direct(3.6) + f_indirect(1.44) = 5.04 per Otto & Steenbuck (2025), Eq. 3.19.
        # CAPEX [€] = f_installation × base_cost_elec × total_area
        self.f_installation_elec = pyo.Param(initialize=ElectrolyzerParameters.f_installation_elec)

        # Annual maintenance cost as fraction of CAPEX [—] — from Economics
        self.maint_factor = pyo.Param(initialize=Economics.maintenance_factor_static)

        # Stack replacement at year 10 (same assumption as PEM-WE, Mucci et al. 2023)
        # Effective CAPEX = CAPEX_initial * (1 + (1+r)^-10)
        _r = Economics.interest_rate
        self.stack_replacement_factor = pyo.Param(initialize=1.0 + (1.0 + _r) ** (-10))

    def build_variables(self):

        # Variables

        self.E_appl = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.E_appl_lb, ElectrolyzerVariablesSurrogate.E_appl_ub),
            initialize=ElectrolyzerVariablesSurrogate.E_appl,
        )  # unit: V

        self.Vcell = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.Vcell_lb, ElectrolyzerVariablesSurrogate.Vcell_ub),
            initialize=ElectrolyzerVariablesSurrogate.Vcell,
        )  # unit: V

        self.eta_actA = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.eta_actA_lb, ElectrolyzerVariablesSurrogate.eta_actA_ub),
            initialize=ElectrolyzerVariablesSurrogate.eta_actA,
        )  # unit: V
        self.eta_ohm = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.eta_ohm_lb, ElectrolyzerVariablesSurrogate.eta_ohm_ub),
            initialize=ElectrolyzerVariablesSurrogate.eta_ohm,
        )  # unit: V
        self.CD = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.CD_lb, ElectrolyzerVariablesSurrogate.CD_ub),
            initialize=ElectrolyzerVariablesSurrogate.CD,
        )  # unit: A/m2
        self.v = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.v_lb, ElectrolyzerVariablesSurrogate.v_ub),
            initialize=ElectrolyzerVariablesSurrogate.v,
        )  # unit: m/s

        self.CO2share = pyo.Var(
            bounds=(ElectrolyzerVariablesSurrogate.CO2share_lb, ElectrolyzerVariablesSurrogate.CO2share_ub),
            initialize=ElectrolyzerVariablesSurrogate.CO2share,
        )  # dimensionless

        # Concentration Variables
        self.c_elec_in_CO2 = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3
        self.c_elec_in_CO = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3
        self.c_elec_out_CO2 = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3
        self.c_elec_out_CO = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3
        self.c_elec_out_H2 = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3
        self.c_elec_mean_CO2 = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/m3

        # Mole flow Variables
        self.nf_CO2_elec_in = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/s
        self.nf_CO_elec_in = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/s
        self.nf_CO2_elec_out = pyo.Var(initialize=100, bounds=(1e-3, 1e6))  # unit: mol/s
        self.nf_CO_elec_out = pyo.Var(initialize=50, bounds=(1e-3, 1e6))  # unit: mol/s
        self.nf_H2_elec_in = pyo.Var(
            initialize=0.0, bounds=(0.0, 1e4), doc="H2 entering from recycle — inert pass-through [mol/s]"
        )
        self.nf_H2_elec_out = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/s

        self.vol_gas_flow = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: m3/s
        self.I = pyo.Var(initialize=1, bounds=(1e-6, 1e12))  # unit: A
        self.Power = pyo.Var(initialize=1, bounds=(1e-6, 1e12))  # unit: W
        self.OPEX = pyo.Var(initialize=1, bounds=(1e-6, 1e12))  # unit: €/h
        self.CAPEX = pyo.Var(initialize=1, bounds=(1e-6, 1e12))  # unit: €

        self.total_area = pyo.Var(
            initialize=ElectrolyzerVariablesSurrogate.total_area, bounds=(3000, 100000)
        )  # unit: m2
        self.n = pyo.Var(
            initialize=ElectrolyzerVariablesSurrogate.n / 1e6, bounds=(3.0, None)
        )  # unit: millions of channels

    def build_constraints(self):
        # Constraints
        # Surrogate Equations
        self.surrogateCO2_out = pyo.Constraint(
            expr=self.c_elec_out_CO2
            == ((self.E_appl * (((sqrt(sqrt(self.v)) + self.E_appl) * -3.7203176) - 1.3941393)) ** 3 + 40.244705)
            * self.CO2share
        )

        self.surrogateCO_out = pyo.Constraint(
            expr=self.c_elec_out_CO
            == self.CO2share
            * (
                (
                    ((1.4358697 / (self.E_appl - (self.v**-0.46066585))) - ((self.E_appl * -0.6278232) ** 21.189863))
                    + (self.E_appl * -1.1215197)
                )
                ** 10.42661
            )
        )

        self.surrogateH2_out = pyo.Constraint(
            expr=self.c_elec_out_H2 == (((self.E_appl * (self.E_appl * 0.26091927)) + 0.35308436) ** 14.994374) / self.v
        )

        # Voltage Calculation
        # Vcell = (anode potential) - (cathode potential) + ohmic, all vs RHE.
        # E_appl is the applied cathode potential vs RHE (= Phi_s in Kas et al. 2021),
        # so (-E_appl) already accounts for the full cathode side including its
        # equilibrium potential. Adding BV_ECO (|E_c0|) here double-counted the CO
        # standard potential (~0.11 V); it is therefore omitted.
        self.CO2Voltage = pyo.Constraint(expr=self.Vcell == self.E_anode + self.eta_actA - self.E_appl + self.eta_ohm)
        self.eta_actA_def = pyo.Constraint(
            expr=self.eta_actA == Constants.R * Constants.T / (0.5 * Constants.F) * asinh(self.CD / (2 * 1e-4))
        )

        self.eta_ohm_def = pyo.Constraint(
            expr=self.eta_ohm == self.CD * (self.Channel_H / self.sigma_el + self.Lm / self.sigma_m)
        )
        self.CD_def = pyo.Constraint(
            expr=self.CD
            == self.BV_iCO
            * self.c_elec_mean_CO2
            / self.BV_CO2ref
            * exp(-(self.BV_CO * Constants.F / (Constants.R * Constants.T) * (self.E_appl + self.BV_ECO)))
            + self.BV_iH2 * exp(-(self.BV_H2 * Constants.F / (Constants.R * Constants.T) * self.E_appl))
        )

        # Mean CO2 concentration (linear interpolation between inlet and outlet)
        CO2_henry = 0.85
        self.c_elec_mean_CO2_constraint = pyo.Constraint(
            expr=self.c_elec_mean_CO2 == (self.c_elec_in_CO2 + self.c_elec_out_CO2) / 2 * CO2_henry
        )

        self.vol_gas_flow_constraint = pyo.Constraint(
            expr=self.vol_gas_flow == self.v * self.Channel_H * self.Channel_W * self.n * 1e6
        )

        # Mole Flow conversions output
        self.nf_CO_elec_out_constraint = pyo.Constraint(
            expr=self.nf_CO_elec_out == self.c_elec_out_CO * self.vol_gas_flow
        )

        self.nf_CO2_elec_out_constraint = pyo.Constraint(
            expr=self.nf_CO2_elec_out == self.c_elec_out_CO2 * self.vol_gas_flow
        )

        self.nf_H2_elec_out_constraint = pyo.Constraint(
            expr=self.nf_H2_elec_out == self.c_elec_out_H2 * self.vol_gas_flow + self.nf_H2_elec_in
        )

        # Mole Flow conversions input
        # self.nf_H2_in = pyo.Constraint(expr=self.nf_H2_in == 0)
        self.nf_CO2_elec_in_constraint = pyo.Constraint(
            expr=self.nf_CO2_elec_in == self.c_elec_in_CO2 * self.vol_gas_flow
        )
        self.nf_CO_elec_in_constraint = pyo.Constraint(expr=self.nf_CO_elec_in == self.c_elec_in_CO * self.vol_gas_flow)

        self.c_in_CO2_constraint = pyo.Constraint(
            expr=self.c_elec_in_CO2 == ((Constants.P / (Constants.R * Constants.T))) * self.CO2share
        )
        self.c_in_CO_constraint = pyo.Constraint(
            expr=self.c_elec_in_CO == ((Constants.P / (Constants.R * Constants.T))) * (1 - self.CO2share)
        )

        self.I_constraint = pyo.Constraint(expr=self.I == self.CD * self.total_area)  # CD in A/m2, area in m2, I in A
        self.Power_constraint = pyo.Constraint(expr=self.Power == self.I * self.Vcell)  # Power in W

        self.opex_constraint = pyo.Constraint(
            expr=self.OPEX
            == self.Power * Economics.price_power / 1000
            + self.maint_factor * self.CAPEX / Economics.operating_hours_per_year
        )  # Power in W, price in €/kWh; maintenance spread over operating hours → €/h

        self.capex_constraint = pyo.Constraint(
            expr=self.CAPEX
            == self.f_installation_elec * self.base_cost_elec * self.total_area * self.stack_replacement_factor
        )  # includes discounted stack replacement at year 10

        self.total_area_constraint = pyo.Constraint(expr=self.total_area == self.n * 1e6 * self.reacting_area)

        # Constraints to keep physics somewhat reasonable.
        # CO should increase (Outlet - Inlet >= 0)
        self.CO_ineq = pyo.Constraint(expr=self.c_elec_out_CO - self.c_elec_in_CO >= 0)

        # CO2 should decrease (Inlet - Outlet >= 0)
        self.CO2_ineq = pyo.Constraint(expr=self.c_elec_in_CO2 - self.c_elec_out_CO2 >= 0)

        # Moved from calculate_performance_metrics to provide lower bound
        self.FE = pyo.Var(initialize=1, bounds=(1e-6, 100))  # unit: %
        self.n_CO2_theoretical = pyo.Var(initialize=1, bounds=(1e-6, 1e6))  # unit: mol/s
        self.lamda_CO2 = pyo.Var(initialize=2, bounds=(1, 20))  # unit: dimensionless

        self.FE_constraint = pyo.Constraint(
            expr=self.FE
            == (
                self.BV_iCO
                * self.c_elec_out_CO2
                * CO2_henry
                / self.BV_CO2ref
                * exp(-1 * (self.BV_CO * Constants.F / (Constants.R * Constants.T) * (self.E_appl + self.BV_ECO)))
                / self.CD
                * 100
            )
        )

        self.n_CO2_theoretical_constraint = pyo.Constraint(expr=self.n_CO2_theoretical == self.I / (2 * Constants.F))
        self.lamda_CO2_constraint = pyo.Constraint(expr=self.lamda_CO2 == self.nf_CO2_elec_in / self.n_CO2_theoretical)

        self.FE_constraint.deactivate()

    def calculate_performance_metrics(self):
        CO2_henry = 0.85

        # These equations are taken from bagemihl and they usually dont work
        self.X_tot = (self.c_elec_in_CO2.value - self.c_elec_out_CO2.value) / self.c_elec_in_CO2.value * 100
        self.X_het = (self.c_elec_out_CO.value * self.BV_CO2m) / self.c_elec_in_CO2.value * 100
        self.X_hom = self.X_tot - self.X_het
        from math import exp as _exp

        self.FE.set_value(
            self.BV_iCO.value
            * self.c_elec_out_CO2.value
            * CO2_henry
            / self.BV_CO2ref.value
            * _exp(
                -1
                * (
                    self.BV_CO.value
                    * Constants.F
                    / (Constants.R * Constants.T)
                    * (self.E_appl.value + self.BV_ECO.value)
                )
            )
            / self.CD.value
            * 100
        )
        # self.FE = (
        #    self.BV_iCO.value
        #    * self.c_elec_out_CO2.value
        #    * CO2_henry
        #    / self.BV_CO2ref.value
        #    * exp(
        #        -1
        #        * (
        #            self.BV_CO.value
        #            * Constants.F
        #            / (Constants.R * Constants.T)
        #            * (self.E_appl.value + self.BV_ECO.value)
        #        )
        #    )
        #    / self.CD.value
        #    * 100
        # )

        # n_CO2_theoretical = self.I.value / (2 * Constants.F)
        # self.lamda_CO2 = self.nf_CO2_elec_in.value / n_CO2_theoretical
        # self.lamda_CO2_temp = self.nf_CO2_elec_in.value / (
        #    self.nf_CO2_elec_in.value - self.nf_CO2_elec_out.value + 1e-6
        # )

    def print_Cmolebalance(self):
        # TODO fix this is wrong doesnt consider electrolyte in molebalance, also add Hmole balance
        X_tot = (self.c_in_CO2.value - self.c_out_CO2.value) / self.c_in_CO2.value
        input_moles = self.nf_CO2_in.value + self.nf_CO_in.value
        output_moles = self.nf_CO2_out.value * (1 + X_tot)
        difference = input_moles - output_moles
        percent_difference = difference / input_moles * 100
        print("CO2 Electrolyser Mole Balance Check")
        print(f"Input Moles: {input_moles} mol/s")
        print(f"Output Moles: {output_moles} mol/s")
        print(f"Difference: {difference} mol/s")
        print(f"Percent Difference: {percent_difference}%")

    def print_variables(self):
        self.calculate_performance_metrics()

        print("Surrogate CO2 Electrolyser")
        table_elec = PrettyTable()
        table_elec.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
        table_elec.add_row(["OPEX", self.OPEX.value, "€/h"])
        table_elec.add_row(["CAPEX", self.CAPEX.value, "€"])
        table_elec.add_row(["Vcell", self.Vcell.value, "V"])
        table_elec.add_row(["I", self.I.value, "A"])
        table_elec.add_row(["Power", self.Power.value / 1e6, "MW"])
        table_elec.add_row(["E_appl", self.E_appl.value, "V"])
        table_elec.add_row(["CD", self.CD.value, "A/m2"])
        table_elec.add_row(["CD", self.CD.value * 0.1, "mA/cm2"])

        table_elec.add_row(["eta_actA", self.eta_actA.value, "V"])
        table_elec.add_row(["eta_ohm", self.eta_ohm.value, "V"])

        table_elec.add_row(["v", self.v.value, "m/s"])
        table_elec.add_row(["vol_gas_flow", self.vol_gas_flow.value, "m3/s"])
        table_elec.add_row(["E_anode", self.E_anode.value, "V"])
        table_elec.add_row(["CO2share", self.CO2share.value, "dimensionless"])
        table_elec.add_row(["c_in_CO2", self.c_elec_in_CO2.value, "mol/m3"])
        table_elec.add_row(["c_in_CO", self.c_elec_in_CO.value, "mol/m3"])
        table_elec.add_row(["c_out_CO2", self.c_elec_out_CO2.value, "mol/m3"])
        table_elec.add_row(["c_out_CO", self.c_elec_out_CO.value, "mol/m3"])
        table_elec.add_row(["c_out_H2", self.c_elec_out_H2.value, "mol/m3"])
        table_elec.add_row(["nf_CO2_in", self.nf_CO2_elec_in.value, "mol/s"])
        table_elec.add_row(["nf_CO_in", self.nf_CO_elec_in.value, "mol/s"])
        table_elec.add_row(["nf_H2_in", self.nf_H2_elec_in.value, "mol/s"])
        table_elec.add_row(["nf_CO2_out", self.nf_CO2_elec_out.value, "mol/s"])
        table_elec.add_row(["nf_CO_out", self.nf_CO_elec_out.value, "mol/s"])
        table_elec.add_row(["nf_H2_out", self.nf_H2_elec_out.value, "mol/s"])

        table_elec.add_row(["total_area", self.total_area.value, "m2"])
        table_elec.add_row(["n", self.n.value * 1e6, "dimensionless"])

        table_elec.add_row(["X_tot", self.X_tot, "%"])
        table_elec.add_row(["X_het", self.X_het, "%"])
        table_elec.add_row(["X_hom", self.X_hom, "%"])
        table_elec.add_row(["FE", self.FE.value, "%"])
        table_elec.add_row(["lamda_CO2", self.lamda_CO2.value, "-"])

        table_elec.add_row(["c_mean_CO2", self.c_elec_mean_CO2.value, "mol/m3"])

        print(table_elec)
