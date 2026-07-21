import pyomo.environ as pyo
from pyomo.environ import units as u
from configuration import Economics, ElectrolyzerParameters, ElectrolyzerVariables, Constants
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
        self.current_density = pyo.Param(initialize=ElectrolyzerParameters.current_density, units=u.A / u.m**2)
        self.voltage = pyo.Param(initialize=ElectrolyzerParameters.voltage, units=u.V)
        self.area_elec = pyo.Param(initialize=ElectrolyzerParameters.area_elec, units=u.m**2)

        self.current = pyo.Param(initialize=ElectrolyzerParameters.current, units=u.A)
        self.power_elec = pyo.Param(initialize=ElectrolyzerParameters.power_elec, units=u.kW)
        self.op_cost_elec = pyo.Param(initialize=ElectrolyzerParameters.op_cost_elec, units=1 / u.h)  # EUR/h

        # Capital Cost (Peters-Timmerhaus-West-Methode)
        self.base_cost_elec = pyo.Param(initialize=ElectrolyzerParameters.base_cost_elec, units=1 / u.kW)  # USD/kW
        self.direct_cost_factor_elec = pyo.Param(initialize=ElectrolyzerParameters.direct_cost_factor_elec)
        self.indirect_cost_factor_elec = pyo.Param(initialize=ElectrolyzerParameters.indirect_cost_factor_elec)
        self.fixed_cost_elec = pyo.Param(initialize=ElectrolyzerParameters.fixed_cost_elec)

    def build_variables(self):

        # Variables
        self.nf_CO2_elec_in = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_CO2_elec_in_lb, ElectrolyzerVariables.nf_CO2_elec_in_ub),
            initialize=ElectrolyzerVariables.nf_CO2_elec_in,
        )  # unit: mol/s
        self.nf_CO_elec_in = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_CO_elec_in_lb, ElectrolyzerVariables.nf_CO_elec_in_ub),
            initialize=ElectrolyzerVariables.nf_CO_elec_in,
        )  # unit: mol/s
        self.nf_CO2_elec_out = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_CO2_elec_out_lb, ElectrolyzerVariables.nf_CO2_elec_out_ub),
            initialize=ElectrolyzerVariables.nf_CO2_elec_out,
        )  # unit: mol/s
        self.nf_CO_elec_out = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_CO_elec_out_lb, ElectrolyzerVariables.nf_CO_elec_out_ub),
            initialize=ElectrolyzerVariables.nf_CO_elec_out,
        )  # unit: mol/s
        self.nf_CO_elec = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_CO_elec_lb, ElectrolyzerVariables.nf_CO_elec_ub),
            initialize=ElectrolyzerVariables.nf_CO_elec,
        )  # unit: mol/s
        self.nf_H2_elec_out = pyo.Var(
            bounds=(ElectrolyzerVariables.nf_H2_elec_out_lb, ElectrolyzerVariables.nf_H2_elec_out_ub),
            initialize=ElectrolyzerVariables.nf_H2_elec_out,
        )  # unit: mol/s

        self.lamda = pyo.Var(
            bounds=(ElectrolyzerVariables.lambda_lb, ElectrolyzerVariables.lambda_ub), initialize=2
        )  # dimensionless
        self.FE_CO = pyo.Var(
            bounds=(ElectrolyzerVariables.FE_CO_lb, ElectrolyzerVariables.FE_CO_ub), initialize=0.9
        )  # dimensionless
        self.FE_H2 = pyo.Var(
            bounds=(ElectrolyzerVariables.FE_H2_lb, ElectrolyzerVariables.FE_H2_ub), initialize=0.1
        )  # dimensionless

    def build_constraints(self):
        # Constraints
        # Electrical Equation
        self.current_const1 = pyo.Constraint(
            expr=self.nf_CO_elec == self.current * self.FE_CO / (Constants.z_CO * Constants.F)
        )
        self.current_const2 = pyo.Constraint(
            expr=self.nf_H2_elec_out == self.current * self.FE_H2 / (Constants.z_H2 * Constants.F)
        )
        self.FE_CO_const = pyo.Constraint(expr=self.FE_CO == (self.lamda**2) / ((self.lamda**2) + 1.5))
        self.FE_H2_const = pyo.Constraint(
            expr=self.FE_H2 == -0.0003 * (self.lamda**3) + 0.0102 * (self.lamda**2) - 0.0995 * self.lamda + 0.3129
        )
        # Material Balances (C-balance)
        self.lamda_definition = pyo.Constraint(expr=self.nf_CO2_elec_in == self.lamda * self.nf_CO_elec)
        self.balance1 = pyo.Constraint(
            expr=self.nf_CO2_elec_in + self.nf_CO_elec_in == self.nf_CO2_elec_out + self.nf_CO_elec_out
        )
        self.balance2 = pyo.Constraint(expr=self.nf_CO2_elec_out <= self.nf_CO2_elec_in)
        self.balance3 = pyo.Constraint(expr=self.nf_CO2_elec_out == self.nf_CO2_elec_in - self.nf_CO_elec)
        self.balance4 = pyo.Constraint(expr=self.nf_CO_elec_out == self.nf_CO_elec_in + self.nf_CO_elec)
    