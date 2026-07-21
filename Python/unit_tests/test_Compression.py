import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.CompressionFlash.CompressionFlowsheet import CompressionFlowsheet


def test_compression_flowsheet():
    z_aspen = {"CO2": 0.68, "CO": 0.26, "H2": 0.06}
    feed_flow = 81.04

    flowsheet = CompressionFlowsheet(
        z_dict=z_aspen,
        feed_flow=feed_flow,
        Pin=1.0,
        Tin=298.15,    # 25 °C
        Pout=30.0,
        T_inter=303.15,
        T_final=238.15,
    )

    flowsheet.solve()
    flowsheet.report()


if __name__ == "__main__":
    test_compression_flowsheet()
