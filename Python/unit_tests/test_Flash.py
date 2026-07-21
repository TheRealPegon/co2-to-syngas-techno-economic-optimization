import os
import sys
import pyomo.environ as pyo
from pyomo.environ import Constraint
from pyomo.environ import value, Constraint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

# Now import the module
from SubModels.CompressionFlash.FlashModel import FlashModel
from SubModels.PSAModel import PSA
from Python.graveyard.flowsheet_builder import FlowsheetModel
from SubModels.CompressionFlash.CompressionFlashFlowsheet import CompressionFlashFlowsheet


def test_flash():
    # --- Implementation ---
    z_aspen = {"CO2": 0.68, "CO": 0.26, "H2": 0.06}
    K_aspen = {"CO2": 0.48, "CO": 20.43, "H2": 88.6}
    feed_flow = 81.04

    # Create the object
    flash = FlashModel(z_aspen, K_aspen, feed_flow, 30, 273.15)

    # Solve it
    flash.solve()

    # See the results
    flash.display_results()

def test_compressor():
    pass


def test_compression_flash_flowsheet():
    z_aspen = {"CO2": 0.68, "CO": 0.26, "H2": 0.06}
    K_aspen = {"CO2": 0.48, "CO": 20.43, "H2": 88.6}
    feed_flow = 81.04

    # Feed starts at 1 bar, 25 °C — same composition/flow as test_flash
    flowsheet = CompressionFlashFlowsheet(
        z_dict=z_aspen,
        K_dict=K_aspen,
        feed_flow=feed_flow,
        Pin=1.0,
        Tin=298.15,    # 25 °C
        Pout=30.0,
        T_flash=273.15,
    )

    flowsheet.solve()
    flowsheet.report()


if __name__ == "__main__":
    test_compression_flash_flowsheet()
