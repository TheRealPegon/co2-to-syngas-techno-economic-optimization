import pyomo.environ as pyo
from pyomo.environ import units as u
from configuration import Economics, H2Parameters, Constants
from math import log, pi
from prettytable import PrettyTable


class H2Model(pyo.ConcreteModel):
    """
    Pyomo model for a PEM/alkaline water electrolyzer producing green hydrogen.

    Models the techno-economics of an electrolyzer plant, including:
    - CAPEX as a function of plant size via a cubic polynomial correlation
      (Reksten et al., Int. J. Hydrogen Energy 47, 2022, 38106-38113), scaled
      by the CE PCI cost index and a conservative capacity factor (C_F_el).
    - OPEX from annual electricity consumption at the given electricity price.
    - Annual maintenance cost as a fixed fraction of CAPEX (membrane equipment rate).
    - Levelized cost of hydrogen (EUR/kg) as annualized total cost divided by
      annual H2 production.

    Decision variable:
        power_hp_mw : Electrolyzer rated power [MW], lower bound 10 MW.

    Constraint structure
    --------------------
    ``build_constraints()`` — core constraints needed for any use of this model
    (CAPEX, OPEX, MAINTENANCE, and H2 output flows).

    ``build_reporting_constraints()`` — additional constraints that compute
    lifetime throughput and the levelized cost of hydrogen.  These are built
    automatically in ``__init__`` for standalone use.  When embedding this model
    in a larger NLP (e.g. ``CombinedFlowsheet``) they should be deactivated to
    avoid introducing bilinear/division terms into the combined problem.

    Typical standalone use::

        model = H2Model()
        solver = pyo.SolverFactory("ipopt")
        solver.solve(model)
        model.print_variables()

    Embedded use (inside CombinedFlowsheet)::

        h2_elec = H2Model()
        for c in ("total_h2_kwh_in_constraint", "total_h2_kg_constraint",
                  "capex_for_1kg_h2_constraint", "annualized_capex_per_kg_h2_constraint",
                  "total_annualized_cost_constraint", "cost_per_kg_h2_constraint"):
            getattr(h2_elec, c).deactivate()
    """

    def __init__(self):
        super().__init__()

        self.build_parameters()
        self.build_variables()
        self.build_constraints()
        self.build_reporting_constraints()

    def build_parameters(self):
        """
        Declare all fixed Pyomo parameters from configuration dataclasses.

        Parameters
        ----------
        C_F_el : float
            Conservative cost factor for electrolyzer CAPEX (dimensionless).
        CE_PCI : float
            CE Plant Cost Index value for 2023, used to scale the reference
            CAPEX correlation (base year index = 708.8).
        electrolyzer_efficiency : float
            Overall electrical efficiency of the electrolyzer (dimensionless).
        plant_life_years : int
            Expected plant lifetime in years, used for annualization.
        operating_hours_per_year : float
            Annual operating hours [h/year].
        actual_kwh_per_kg_h2 : float
            Actual electrical energy consumption [kWh/kg H2], derived from
            the HHV of H2 divided by electrolyzer efficiency.
        annuity_factor_plant_life : float
            Net present value annuity factor over the plant lifetime at the
            given discount rate (sum of discount factors).
        price_power : float
            Electricity purchase price [EUR/kWh].
        MW_H2 : float
            Molar mass of hydrogen [kg/mol].
        maintenance_factor : float
            Annual maintenance cost as a fraction of installed CAPEX [1/year],
            using the membrane equipment rate from Economics.
        """
        self.economics = Economics()
        self.C_F_el = pyo.Param(initialize=H2Parameters.C_F_el)
        self.CE_PCI = pyo.Param(initialize=Economics.CEPCI_2024)
        self.electrolyzer_efficiency = pyo.Param(initialize=H2Parameters.electrolyzer_efficiency)
        self.plant_life_years = pyo.Param(initialize=self.economics.plant_life_years)
        self.operating_hours_per_year = pyo.Param(initialize=self.economics.operating_hours_per_year)
        self.actual_kwh_per_kg_h2 = pyo.Param(initialize=H2Parameters.actual_kwh_per_kg_h2)
        self.annuity_factor_plant_life = pyo.Param(initialize=self.economics.annuity_factor_plant_life)
        self.price_power = pyo.Param(initialize=self.economics.price_power)
        self.MW_H2 = pyo.Param(initialize=Constants.MW_H2)
        self.maintenance_factor = pyo.Param(initialize=self.economics.maintenance_factor_membrane)
        # PEM stack replacement at year 10 (lifetime ~30-90 kh, Mucci et al. 2023)
        # Effective CAPEX = CAPEX_initial * (1 + (1+r)^-10)
        _r = self.economics.interest_rate
        self.stack_replacement_factor = pyo.Param(initialize=1.0 + (1.0 + _r) ** (-10))

    def build_variables(self):
        """
        Declare all Pyomo decision and intermediate variables.

        Core variables (used in both standalone and embedded mode)
        ----------------------------------------------------------
        power_hp_mw : float
            Electrolyzer rated power [MW]. Primary decision variable; lb = 10 MW.
        x_hp : float
            Natural log of power_hp_mw [-]. Auxiliary variable to allow the
            cubic CAPEX polynomial to be expressed in log-space for the solver.
        capex_electrolyzer_per_kw : float
            Specific CAPEX of the electrolyzer [EUR/kW].
        CAPEX : float
            Total installed CAPEX of the electrolyzer [EUR].
        OPEX : float
            Annual operating cost from electricity consumption [EUR/year].
        MAINTENANCE : float
            Annual maintenance cost as a fraction of CAPEX [EUR/year].
        H2_out_kg_per_sec : float
            Continuous H2 production rate [kg/s].
        H2_out_mol_per_sec : float
            Continuous H2 production rate [mol/s].

        Reporting variables (constrained by :meth:`build_reporting_constraints`)
        -------------------------------------------------------------------------
        total_h2_kwh_in : float
            Total electrical energy consumed over the full plant lifetime [kWh].
        total_h2_kg : float
            Total mass of H2 produced over the full plant lifetime [kg].
        capex_for_1kg_h2 : float
            Lifetime CAPEX allocated per kg of H2 produced [EUR/kg].
        annualized_capex_per_kg_h2 : float
            Annualized CAPEX per kg of H2, derived from capex_for_1kg_h2
            divided by the annuity factor [EUR/kg].
        total_annualized_cost : float
            Total annualized cost (annualized CAPEX + OPEX + MAINTENANCE) [EUR/year].
        cost_per_kg_h2 : float
            Levelized cost of hydrogen [EUR/kg].
        """
        self.power_hp_mw = pyo.Var(bounds=(10, None), initialize=10)
        self.x_hp = pyo.Var(bounds=(0, 10), initialize=7.47712125472)
        self.capex_electrolyzer_per_kw = pyo.Var()
        self.CAPEX = pyo.Var()
        self.total_h2_kwh_in = pyo.Var(initialize=1)
        self.total_h2_kg = pyo.Var(initialize=1)
        self.capex_for_1kg_h2 = pyo.Var()
        self.annualized_capex_per_kg_h2 = pyo.Var()  # Euro/year
        self.OPEX = pyo.Var()  # Euro/year
        self.MAINTENANCE = pyo.Var(initialize=0)  # Euro/year

        self.H2_out_kg_per_sec = pyo.Var(initialize=1)  # kg/sec
        self.H2_out_mol_per_sec = pyo.Var(initialize=1)  # mol/sec
        self.total_annualized_cost = pyo.Var()  # Euro/year
        self.cost_per_kg_h2 = pyo.Var()  # Euro/kg

    def build_constraints(self):
        """
        Declare all Pyomo equality constraints linking parameters and variables.

        CAPEX constraints
        -----------------
        x_hp_constraint :
            Defines x_hp = log(power_hp_mw). Introduces the log-scale auxiliary
            variable used by the polynomial CAPEX correlation.

        capex_electrolyzer_per_kw_constraint :
            Cubic polynomial fit of specific CAPEX vs. log(plant size):
                CAPEX [EUR/kW] = (-38.131*x^3 + 433.69*x^2 - 1690*x + 3634.6)
                                 * (CE_PCI / 708.8) * C_F_el
            Source: Reksten et al., Int. J. Hydrogen Energy 47 (2022) 38106-38113.
            The CE PCI ratio escalates costs from the reference year to 2023.

        capex_constraint :
            Total CAPEX [EUR] = capex_electrolyzer_per_kw * power_hp_mw * 1000.

        Production constraints
        ----------------------
        H2_out_kg_per_sec_constraint :
            Instantaneous H2 flow [kg/s] = power_hp_mw * 1000
            / actual_kwh_per_kg_h2 / 3600.

        H2_out_mol_per_sec_constraint :
            Instantaneous H2 flow [mol/s] = H2_out_kg_per_sec / MW_H2.

        Cost constraints
        ----------------
        opex_constraint :
            Annual electricity cost [EUR/year] = operating_hours_per_year
            * power_hp_mw * price_power * 1000.

        maintenance_constraint :
            Annual maintenance cost [EUR/year] = maintenance_factor * CAPEX.

        Notes
        -----
        Reporting constraints (lifetime throughput and levelized cost) are built
        separately in :meth:`build_reporting_constraints`, which is called
        automatically in ``__init__``.  When this model is embedded in a larger
        NLP (e.g. ``CombinedFlowsheet``) those constraints can be deactivated to
        remove bilinear/division terms that are not needed by the combined
        objective.
        """
        # --- Electrolyzer CAPEX Calculation ---

        self.x_hp_constraint = pyo.Constraint(expr=self.x_hp == pyo.log(self.power_hp_mw))

        # Correlation: CAPEX_ele_HP = -38.131*x_HP^3 + 433.69*x_HP^2 - 1690*x_HP + 3634.6
        # This is the linearized version of a model from Simone paper which cited Anita H. Reksten, Magnus S. Thomassen, Steffen Møller-Holst, Kyrre Sundseth,
        # Projecting the future cost of PEM and alkaline water electrolysers; a CAPEX
        # model including electrolyser plant size and technology development, Int. J.
        # Hydrogen Energy 47 (2022) 38106–38113,
        self.capex_electrolyzer_per_kw_constraint = pyo.Constraint(
            expr=self.capex_electrolyzer_per_kw
            == (-38.131 * (self.x_hp**3) + 433.69 * (self.x_hp**2) - 1690 * self.x_hp + 3634.6)
            * (self.CE_PCI / 708.8 * (self.C_F_el))
        )

        self.capex_constraint = pyo.Constraint(
            expr=self.CAPEX
            == self.capex_electrolyzer_per_kw * self.power_hp_mw * 1000 * self.stack_replacement_factor
        )  # € — includes discounted stack replacement at year 10

        self.H2_out_kg_per_sec_constraint = pyo.Constraint(
            expr=self.H2_out_kg_per_sec == self.power_hp_mw * 1000 / self.actual_kwh_per_kg_h2 / 3600
        )  # kg/s
        self.H2_out_mol_per_sec_constraint = pyo.Constraint(
            expr=self.H2_out_mol_per_sec == self.H2_out_kg_per_sec / self.MW_H2
        )  # mol/s

        self.opex_constraint = pyo.Constraint(
            expr=self.OPEX == self.operating_hours_per_year * self.power_hp_mw * self.price_power * 1000
        )  # €/year

        self.maintenance_constraint = pyo.Constraint(
            expr=self.MAINTENANCE == self.maintenance_factor * self.CAPEX
        )  # €/year

    def build_reporting_constraints(self):
        """Declare constraints used only for reporting the levelized cost of hydrogen.

        These constraints compute lifetime throughput and LCOH metrics that are
        not required by the optimizer when this model is embedded in a larger
        flowsheet.  They are built automatically by ``__init__`` for standalone
        use, but can be deactivated en-bloc when embedding to avoid adding
        bilinear/division terms to the combined NLP.

        Constraints
        -----------
        total_h2_kwh_in_constraint :
            Lifetime electricity input [kWh] = power_hp_mw * 1000
            * operating_hours_per_year * plant_life_years.

        total_h2_kg_constraint :
            Lifetime H2 produced [kg] = total_h2_kwh_in / actual_kwh_per_kg_h2.

        capex_for_1kg_h2_constraint :
            Lifetime CAPEX per kg H2 [EUR/kg] = CAPEX / total_h2_kg.

        annualized_capex_per_kg_h2_constraint :
            Annualized CAPEX per kg H2 [EUR/kg] = capex_for_1kg_h2
            / annuity_factor_plant_life.

        total_annualized_cost_constraint :
            Total annualized cost [EUR/year] = annualized CAPEX contribution
            + OPEX + MAINTENANCE.

        cost_per_kg_h2_constraint :
            Levelized cost of hydrogen [EUR/kg] = total_annualized_cost
            / (H2_out_kg_per_sec * 3600 * operating_hours_per_year).
        """
        self.total_h2_kwh_in_constraint = pyo.Constraint(
            expr=self.total_h2_kwh_in == self.power_hp_mw * 1000 * self.operating_hours_per_year * self.plant_life_years
        )

        self.total_h2_kg_constraint = pyo.Constraint(
            expr=self.total_h2_kg == self.total_h2_kwh_in / self.actual_kwh_per_kg_h2
        )

        self.capex_for_1kg_h2_constraint = pyo.Constraint(expr=self.capex_for_1kg_h2 == self.CAPEX / self.total_h2_kg)

        # Annualize the total CAPEX. The result is the annualized cost to produce 1 kg of H2.
        self.annualized_capex_per_kg_h2_constraint = pyo.Constraint(
            expr=self.annualized_capex_per_kg_h2 == self.capex_for_1kg_h2 / self.annuity_factor_plant_life
        )

        self.total_annualized_cost_constraint = pyo.Constraint(
            expr=self.total_annualized_cost
            == self.annualized_capex_per_kg_h2 * self.H2_out_kg_per_sec * 3600 * self.operating_hours_per_year
            + self.OPEX
            + self.MAINTENANCE
        )

        self.cost_per_kg_h2_constraint = pyo.Constraint(
            expr=self.cost_per_kg_h2
            == self.total_annualized_cost / (self.H2_out_kg_per_sec * 3600 * self.operating_hours_per_year)
        )

    def print_variables(self):
        """Print a formatted summary of all model variables and their solved values."""

        print("H2 Electrolyser")
        table_elec = PrettyTable()
        table_elec.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
        table_elec.add_row(["H2_out_mol_per_sec", self.H2_out_mol_per_sec.value, "mol/s"])
        table_elec.add_row(["H2_out_kg_per_sec", self.H2_out_kg_per_sec.value, "kg/s"])
        table_elec.add_row(["total_h2_kg", self.total_h2_kg.value, "kg"])
        table_elec.add_row(["total_h2_kwh_in", self.total_h2_kwh_in.value / 1e6, "GWh"])
        table_elec.add_row(["power_hp_mw", self.power_hp_mw.value, "MW"])
        table_elec.add_row(["x_hp", self.x_hp.value, "hp"])
        table_elec.add_row(["actual_kwh_per_kg_h2", self.actual_kwh_per_kg_h2.value, "kWh/kg"])
        print(table_elec)

        table_elec = PrettyTable()
        table_elec.field_names = ["     Variables     ", "                   Values                  ", "  Unit  "]
        table_elec.add_row(["cost_per_kg_h2", self.cost_per_kg_h2.value, "€/kg"])
        table_elec.add_row(["total_annualized_cost", self.total_annualized_cost.value / 1e6, "M€"])

        table_elec.add_row(["OPEX", self.OPEX.value / 1e6, "M€/year"])
        table_elec.add_row(["MAINTENANCE", self.MAINTENANCE.value / 1e6, "M€/year"])
        table_elec.add_row(["CAPEX", self.CAPEX.value / 1e6, "M€"])
        table_elec.add_row(["capex_electrolyzer_per_kw", self.capex_electrolyzer_per_kw.value, "€"])
        table_elec.add_row(["annualized_capex_per_kg_h2", self.annualized_capex_per_kg_h2.value, "€/kg"])
        table_elec.add_row(["capex_for_1kg_h2", self.capex_for_1kg_h2.value, "€/kg"])
        table_elec.add_row(["annuity_factor_plant_life", self.annuity_factor_plant_life.value, ""])
        table_elec.add_row(["plant_life_years", self.plant_life_years.value, ""])
        table_elec.add_row(["operating_hours_per_year", self.operating_hours_per_year.value, ""])
        print(table_elec)
