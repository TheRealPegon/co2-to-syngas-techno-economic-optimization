# Add parent directory to sys.pathimport sys
import os
import sys
import pyomo.environ as pyo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from configuration import Economics, H2Parameters, Constants

# Now import the module
from SubModels.H2_Electrolyzer_Model import H2Model

def test_H2_model():
    model = H2Model()

    syngas_ratio = 2
    syngas_ratio_kg = syngas_ratio * Constants.MW_H2 / (Constants.MW_CO)  # in kg/kg_syngas
    syngas_target_per_day = 100*1e3  # target is 100 tonnes per day
    syngas_target_per_hour = syngas_target_per_day / 24  # in kg/h
    
    h2_target = syngas_target_per_hour * (syngas_ratio_kg / (1 + syngas_ratio_kg)) / 3600  # in kg/s
            
    model.H2_out_kg_per_sec.fix(h2_target)  # Fixing H2 output to the calculated target value for testing  
    #model.power_hp_mw.fix(10)  # Fixing power input to 10 MW for testing
    
    model.test_obj = pyo.Objective(expr=0)

    # Solve the model
    solver = pyo.SolverFactory("ipopt")
    solver.options['halt_on_ampl_error'] = 'yes'
    results = solver.solve(model, tee=True)
    model.print_variables()

if __name__ == "__main__":

    test_H2_model()