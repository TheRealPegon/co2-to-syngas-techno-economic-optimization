import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import the module
from Python.graveyard.CO2_Electrolyzer_Interface import CO2_Electrolyzer_Interface


def test_run_simulation():
    c0 = [1e5 / (8.314 * 298.15), 0, 0]
    results = CO2_Electrolyzer_Interface().run_simulation(-1.0, c0)

    assert results.X["tot"] == 0.0016849020130455215

test_run_simulation()