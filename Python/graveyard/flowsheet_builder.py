import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))
import pyomo.environ as pyo
from SubModels.CO2_ElectrolyzerModel import CO2Electrolyzer
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer as CO2ElectrolyzerSurrogate
from SubModels.PSAModel import PSA
from SubModels.H2_Electrolyzer_Model import H2Model
from configuration import Economics, Constants
from prettytable import PrettyTable


class FlowsheetModel(pyo.ConcreteModel):

    def __init__(
        self,
        CO2Electrolyzer_type="Standard",
        H2_type="Buy",
        syngas_ratio=2,
        syngas_target=75,
        set_target_flow=True,
        purge=0.1,
    ):
        """_summary_

        Args:
            CO2Electrolyzer_type (str, optional): _description_. Defaults to "Standard".
            H2_type (str, optional): _description_. Defaults to "Buy".
            syngas_ratio (int, optional): _description_. Defaults to 2.
            syngas_target (float, optional): tonnes per day. Defaults to 75.
        """

        super().__init__()
        self.CO2Electrolyzer_type = CO2Electrolyzer_type
        self.H2_type = H2_type
        self.syngas_ratio_parameter = syngas_ratio
        self.syngas_target = syngas_target
        self.set_target_flow = set_target_flow
        self.purge = purge
        self.build_flowsheet()

    def minimize_Cost(self, multistart=True):
        if multistart == True:
            opt = pyo.SolverFactory("multistart")
            results = opt.solve(
                self,
                solver="ipopt",
                solver_args={"tee": True},  # Pass tee here instead of directly
            )

            return results

    def build_flowsheet(self):

        if self.CO2Electrolyzer_type == "Standard":
            self.CO2electrolyzer = CO2Electrolyzer()
        elif self.CO2Electrolyzer_type == "Surrogate":
            self.CO2electrolyzer = CO2ElectrolyzerSurrogate()
        self.psa = PSA()
        if self.H2_type == "Electrolyzer":
            self.H2electrolyzer = H2Model()

        self.build_connectors()

        if self.set_target_flow == True:
            self.set_syngas_target()
            self.build_economics()

    def build_connectors(self):
        self.nf_CO2_feed = pyo.Var(
            bounds=(10, 1000),
            initialize=100,
        )  # unit: mol/s

        # Connectors between Electrolyzer and PSA

        # Output of ELectrolyzer = input of PSA, H2 is assumed to be seperated before hand for simplicity
        self.connector1 = pyo.Constraint(expr=self.psa.nf_CO2_psaH_in == self.CO2electrolyzer.nf_CO2_elec_out)
        self.connector2 = pyo.Constraint(expr=self.psa.nf_CO_psaH_in == self.CO2electrolyzer.nf_CO_elec_out)

        # Recycle
        self.connector_recycleCO2 = pyo.Constraint(
            expr=self.CO2electrolyzer.nf_CO2_elec_in == self.nf_CO2_feed + self.psa.nf_CO2_psaL_out * (1 - self.purge)
        )
        self.connector_recycleCO = pyo.Constraint(
            expr=self.CO2electrolyzer.nf_CO_elec_in == self.psa.nf_CO_psaL_out * (1 - self.purge)
        )

    def set_syngas_target(self):
        # ------------Parameters----------------
        self.syngas_ratio = pyo.Param(initialize=self.syngas_ratio_parameter)

        self.mf_syngas = pyo.Param(initialize=self.syngas_target * 1000 / (3600 * 24))  # in kg/s
        self.MW_syngas = pyo.Param(
            initialize=(self.syngas_ratio * Constants.MW_H2 + Constants.MW_CO) / (self.syngas_ratio + 1)
        )  # in kg/mol

        self.MW_H2 = pyo.Param(initialize=Constants.MW_H2)
        self.MW_CO2 = pyo.Param(initialize=Constants.MW_CO2)
        self.MW_CO = pyo.Param(initialize=Constants.MW_CO)

        inital_guess_syngas_flow = self.mf_syngas / self.MW_syngas
        # ------------VALUES----------------

        # implicitly fixed through syngas target and CO2 purity target on the PSA side
        self.nf_CO2_syngas = pyo.Var(initialize=0.1, bounds=(1e-6, 1))  # in mol/s

        self.nf_syngas = pyo.Var(initialize=inital_guess_syngas_flow, bounds=(1e-6, 1e3))  # in mol/s

        self.nf_H2_syngas = pyo.Var(
            initialize=inital_guess_syngas_flow * self.syngas_ratio / (self.syngas_ratio + 1), bounds=(1e-6, 1000)
        )  # in mol/s
        self.nf_CO_syngas = pyo.Var(
            initialize=inital_guess_syngas_flow * 1 / (self.syngas_ratio + 1), bounds=(1e-6, 1e3)
        )  # in mol/s

        # ------------Constraints----------------
        self.nf_syngas_constraint = pyo.Constraint(
            expr=self.nf_syngas == self.nf_H2_syngas + self.nf_CO2_syngas + self.nf_CO_syngas
        )
        self.mf_syngas_constraint = pyo.Constraint(
            expr=self.mf_syngas
            == (self.nf_H2_syngas * self.MW_H2 + self.nf_CO2_syngas * self.MW_CO2 + self.nf_CO_syngas * self.MW_CO)
        )
        self.syngas_ratio_constraint = pyo.Constraint(expr=self.nf_H2_syngas == self.syngas_ratio * self.nf_CO_syngas)

        # Connectors between syngas and Seperation
        self.syngas_sep_connector1 = pyo.Constraint(expr=self.psa.nf_CO2_product == self.nf_CO2_syngas)
        self.syngas_sep_connector2 = pyo.Constraint(expr=self.psa.nf_CO_product == self.nf_CO_syngas)
        if self.H2_type == "Buy":
            self.nf_H2_buy = pyo.Var(initialize=100, bounds=(1e-6, 1e6))  # in mol/s
            self.syngas_sep_connector3 = pyo.Constraint(
                expr=self.nf_H2_syngas == self.CO2electrolyzer.nf_H2_elec_out + self.nf_H2_buy
            )
        elif self.H2_type == "Electrolyzer":
            self.syngas_sep_connector3 = pyo.Constraint(
                expr=self.nf_H2_syngas == self.CO2electrolyzer.nf_H2_elec_out + self.H2electrolyzer.nf_H2_elec_out
            )

    def build_economics(self):
        # Prices
        self.convert_usd_to_euro = pyo.Param(initialize=0.96)
        self.price_CO = pyo.Param(initialize=self.convert_usd_to_euro * 0.81 * 28.01 / 1000)  # in €/mol
        self.price_H2 = pyo.Param(initialize=self.convert_usd_to_euro * 3 * 2.016 / 1000)  # in €/mol
        self.price_CO2 = pyo.Param(initialize=0.05 * 44.01 / 1000)  # in €/mol
        self.price_power = pyo.Param(initialize=0.08)  # in €/kWh

        # Operating Time and Lifetime — sourced from Economics dataclass
        _econ = Economics()
        self.lifetime = pyo.Param(initialize=_econ.plant_life_years)  # in years
        self.operating_day_per_year = pyo.Param(initialize=_econ.operating_days_per_year)  # in days/a
        self.operating_hours_lifetime = pyo.Param(initialize=_econ.plant_life_years * _econ.operating_hours_per_year)  # in h

        # CAPEX
        if self.H2_type == "Buy":
            self.CAPEX = pyo.Var(bounds=(1, 1e12))  # in €
            self.CAPEX_const = pyo.Constraint(expr=self.CAPEX == self.CO2electrolyzer.CAPEX + self.psa.CAPEX)
        elif self.H2_type == "Electrolyzer":
            self.CAPEX = pyo.Var(bounds=(1, None))  # in €
            self.CAPEX_const = pyo.Constraint(
                expr=self.CAPEX == self.CO2electrolyzer.CAPEX + self.psa.CAPEX + self.H2electrolyzer.CAPEX
            )

        # annual OPEX
        seconds_per_year = 3600 * 24 * self.operating_day_per_year
        if self.H2_type == "Buy":
            self.OPEX = pyo.Var(bounds=(0.0001, 1e12))  # in € per year
            self.OPEX_const = pyo.Constraint(
                expr=self.OPEX
                == self.CO2electrolyzer.OPEX
                + self.psa.op_cost_psa * 24 * self.operating_day_per_year
                + (self.price_H2 * self.nf_H2_buy + self.price_CO2 * self.nf_CO2_feed) * seconds_per_year
            )

        elif self.H2_type == "Electrolyzer":
            self.OPEX = pyo.Var(bounds=(0.0001, 1e12))  # in € per year
            self.OPEX_const = pyo.Constraint(
                expr=self.OPEX
                == self.CO2electrolyzer.OPEX
                + self.psa.op_cost_psa * 24 * self.operating_day_per_year
                + self.H2electrolyzer.OPEX
                + (self.price_CO2 * self.nf_CO2_feed) * seconds_per_year
            )

        # TODO: price syngas correctly the 1 stands for 1 euro per kg
        self.revenue = pyo.Var(bounds=(0.0001, 1e12))  # in €/year
        self.revenue_const = pyo.Constraint(
            expr=self.revenue == self.mf_syngas * 1 * 3600 * 24 * self.operating_day_per_year
        )

    def build_objective(self):

        # Objective
        self.objective = pyo.Objective(
            expr=self.CAPEX + self.lifetime * (self.OPEX - self.revenue),
            sense=pyo.minimize,
        )

    def display_results(self):

        if self.CO2Electrolyzer_type == "Surrogate":
            self.CO2electrolyzer.calculate_performance_metrics()
            lamda = self.CO2electrolyzer.lamda_CO2

        elif self.CO2Electrolyzer_type == "Standard":
            lamda = self.CO2electrolyzer.lamda.value

        if self.set_target_flow == True:

            # usually the target flow fixed called self.mf_CO2_syngas does not get hit and the plant underperforms, this is the actual mass flow based on the mole flow that in most situations is correct
            self.actual_mf_syngas = self.MW_syngas.value * self.nf_syngas.value * 1e-3 * 3600 * 24  # in t/day

            print("Entire Plant")
            plant = PrettyTable()
            plant.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
            plant.add_row(["CAPEX ", self.CAPEX.value * 1e-6, "M€"])
            plant.add_row(["OPEX yearly", self.OPEX.value * 1e-6, "M€/year"])
            plant.add_row(["OPEX daily ", self.OPEX.value / self.operating_day_per_year.value, "€/day"])
            plant.add_row(
                [
                    "Syngas production Price ",
                    self.OPEX.value / self.operating_day_per_year.value / self.actual_mf_syngas,
                    "€/t",
                ]
            )
            plant.add_row(["Revenue ", self.revenue.value, "€/year"])  # this is bulllshit for now

            plant.add_row(["-" * 10, "-" * 5, "-" * 10])
            psa_yearly = self.psa.op_cost_psa.value * 24 * self.operating_day_per_year.value
            plant.add_row(["CAPEX CO2 Electrolyzer", self.CO2electrolyzer.CAPEX.value * 1e-6, "M€"])
            plant.add_row(["CAPEX PSA", self.psa.CAPEX.value * 1e-6, "M€"])
            plant.add_row(["OPEX yearly CO2 Electrolyzer", self.CO2electrolyzer.OPEX.value, "€/year"])
            plant.add_row(["OPEX yearly PSA", psa_yearly * 1e-6, "M€/year"])

            plant.add_row(["-" * 10, "-" * 5, "-" * 10])
            plant.add_row(["PSA OPEX cost per t of syngas", self.psa.op_cost_psa.value / self.actual_mf_syngas, "€/t"])
            co_tonnes_per_hour = self.nf_CO_syngas.value * self.MW_CO.value * 1e-3 * 3600
            plant.add_row(["PSA OPEX cost per t/h of CO", self.psa.op_cost_psa.value / co_tonnes_per_hour, "€/(t/h)"])
            plant.add_row(["-" * 10, "-" * 5, "-" * 10])

            plant.add_row(["CO2 Feed ", self.nf_CO2_feed.value, "mol/s"])
            plant.add_row(["Syngas flow ", self.nf_syngas.value, "mol/s"])
            plant.add_row(["Recycled CO2 ", self.psa.nf_CO2_psaL_out.value, "mol/s"])
            plant.add_row(["Recycled CO ", self.psa.nf_CO_psaL_out.value, "mol/s"])

            plant.add_row(["Syngas mass flow target ", self.mf_syngas.value * 1e-3 * 3600 * 24, "t/day"])
            plant.add_row(["Syngas mass flow actual ", self.actual_mf_syngas, "t/day"])
            plant.add_row(["Syngas H2/CO ratio ", self.nf_H2_syngas.value / self.nf_CO_syngas.value, "mol/mol"])
            plant.add_row(["Syngas CO2 content ", self.nf_CO2_syngas.value / self.nf_syngas.value, "mol/mol"])
            plant.add_row(["Syngas CO2 ", self.nf_CO2_syngas.value, "mol/s"])
            plant.add_row(["Syngas CO2 mass flow ", self.nf_CO2_syngas.value * self.MW_CO2.value * 1e-3 * 3600 * 24, "t/day"])
            plant.add_row(["Syngas H2 ", self.nf_H2_syngas.value, "mol/s"])
            plant.add_row(["Syngas H2 mass flow ", self.nf_H2_syngas.value * self.MW_H2.value * 1e-3 * 3600 * 24, "t/day"])
            plant.add_row(["Syngas CO ", self.nf_CO_syngas.value, "mol/s"])
            plant.add_row(["Syngas CO mass flow ", self.nf_CO_syngas.value * self.MW_CO.value * 1e-3 * 3600 * 24, "t/day"])
            plant.add_row(["Purge ", self.purge, ""])

            print(plant)

        self.CO2electrolyzer.print_variables()
        self.psa.print_model()
