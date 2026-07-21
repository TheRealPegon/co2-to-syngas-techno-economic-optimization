# Add parent directory to sys.pathimport sys

# TODO run minimization problem
import os
import sys
import pyomo.environ as pyo
from pyomo.environ import Constraint
from pyomo.environ import value, Constraint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

# Now import the module
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer
from SubModels.PSAModel import PSA
from configuration import ElectrolyzerParametersSurrogate, ElectrolyzerVariablesSurrogate


def test_surrogate_model():
    # Model Completeness Check puts in dummy objective function and fixes enough variables to just arithmetically solve the model
    # with E_appl = -1.31, v = 0.08, CO2share = 0.95, and total_area = 1000, lamda = 3.9586953607704483 (high side of what this model can push for)
    # main infeasability region is if v too high compared to E_appl and low CO2 share -> not CO conversion happening -> inaccuracy in surrogate eq for CO out kicks in -> CO in > CO out -> inequality constraint in model violated
    # example setpoint: v = 0.021, E_appl = -1.31, CO2share = 0.8

    model = CO2Electrolyzer()

    # Set test values for variables
    model.E_appl.fix(-1.31)
    # model.CD.fix(3000)
    model.v.fix(0.08)
    model.CO2share.fix(0.95)
    model.total_area.fix(ElectrolyzerVariablesSurrogate.total_area)

    model.test_obj = pyo.Objective(expr=0)

    # Solve the model
    solver = pyo.SolverFactory("ipopt")
    results = solver.solve(model, tee=True)
    # assert model.nf_CO_out.value == 1.9817696371503047e-07
    # assert model.Vcell.value == 4.103552596263656
    # model.print_Cmolebalance()
    model.print_variables()
    model.display()


def test_surrogate_model_variable_area():
    model = CO2Electrolyzer()

    # Set test values for variables
    model.E_appl.fix(-1.31)
    # model.CD.fix(3000)
    model.v.fix(0.08)
    model.CO2share.fix(0.95)
    # model.nf_CO2_elec_in.fix(100)
    model.nf_CO_elec_out.fix(27.09)

    model.test_obj = pyo.Objective(expr=0)

    # Solve the model
    solver = pyo.SolverFactory("ipopt")
    results = solver.solve(model, tee=True)
    # assert model.nf_CO_out.value == 1.9817696371503047e-07
    # assert model.Vcell.value == 4.103552596263656
    # model.print_Cmolebalance()
    model.print_variables()
    model.display()





if __name__ == "__main__":
    # test_surrogate_model()
    test_surrogate_model_variable_area()

