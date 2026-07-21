import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

import pyomo.environ as pyo
from typing import cast
from configuration import Economics, Constants
from SubModels.CompressionFlash.MultiStageCompressionFlowsheet import MultiStageCompressionFlowsheet

Z = {"CO2": 0.68, "CO": 0.26, "H2": 0.06}
K = {"CO2": 0.48, "CO": 20.43, "H2": 88.6}
FEED = 228
POUT = 19.0  # bar — compression outlet pressure


def test_1stage_no_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=1, with_flash=False,
    )
    fs.solve()
    fs.report()


def test_1stage_with_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=1, with_flash=True, K_dict=K,
    )
    fs.solve()
    fs.report()


def test_2stage_no_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=2, with_flash=False,
    )
    fs.solve()
    fs.report()


def test_2stage_with_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=2, with_flash=True, K_dict=K,
    )
    fs.solve()
    fs.report()


def test_3stage_with_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=3, with_flash=True, K_dict=K,
    )
    fs.solve()
    fs.report()


def test_4stage_with_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=4, with_flash=True, K_dict=K,
    )
    fs.solve()
    fs.report()


def test_5stage_with_flash():
    fs = MultiStageCompressionFlowsheet(
        z_dict=Z, feed_flow=FEED, Pout=POUT,
        n_stages=5, with_flash=True, K_dict=K,
    )
    fs.solve()
    fs.report()


if __name__ == "__main__":
    flash_flowsheets = {}  # n_stages -> solved with-flash flowsheet

    for n in range(1, 6):
        print(f"\n{'#' * 60}")
        print(f"# {n}-stage, no flash")
        fs_no = MultiStageCompressionFlowsheet(
            z_dict=Z, feed_flow=FEED, Pout=POUT, n_stages=n, with_flash=False,
        )
        fs_no.solve()  # type: ignore[operator]
        fs_no.report()  # type: ignore[operator]

        print(f"\n{'#' * 60}")
        print(f"# {n}-stage, with flash")
        fs_fl = MultiStageCompressionFlowsheet(
            z_dict=Z, feed_flow=FEED, Pout=POUT, n_stages=n, with_flash=True, K_dict=K,
        )
        fs_fl.solve()  # type: ignore[operator]
        fs_fl.report()  # type: ignore[operator]
        flash_flowsheets[n] = fs_fl

    fs1, fs2, fs3, fs4, fs5 = (flash_flowsheets[n] for n in range(1, 6))

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    W = 103
    print("\n" + "=" * W)
    print(f"{'COMPARISON SUMMARY':^{W}}")
    print("=" * W)
    print(f"  {'Metric':<35} {'1-stage':>11} {'2-stage':>11} {'3-stage':>11} {'4-stage':>11} {'5-stage':>11}")
    print(f"  {'-'*98}")

    def v(x) -> float:
        return cast(float, pyo.value(x))

    comp_power = [sum((v(c.power_kw) for c in fs._comps), 0.0) for fs in (fs1, fs2, fs3, fs4, fs5)]
    ic_duty    = [sum((v(ic.duty_kw) for ic in fs._ics),  0.0) for fs in (fs1, fs2, fs3, fs4, fs5)]
    hx1_duty = [v(fs.hx1.duty_kw) for fs in (fs1, fs2, fs3, fs4, fs5)]
    hx2_duty = [v(fs.hx2.duty_kw) for fs in (fs1, fs2, fs3, fs4, fs5)]
    capex    = [v(fs.total_capex) for fs in (fs1, fs2, fs3, fs4, fs5)]
    opex     = [v(fs.total_opex)  for fs in (fs1, fs2, fs3, fs4, fs5)]
    lcc      = [v(fs.total_lifecycle_cost) for fs in (fs1, fs2, fs3, fs4, fs5)]
    psi      = [v(fs.flash.psi)   for fs in (fs1, fs2, fs3, fs4, fs5)]

    def row(label, vals, fmt="{:>11.1f}"):
        print(f"  {label:<35} " + " ".join(fmt.format(x) for x in vals))

    # CO2 captured = liquid-phase CO2 flow [mol/s]
    _econ   = Economics()
    MW_CO2  = Constants.MW_CO2                      # 0.04401 kg/mol
    op_hrs  = _econ.operating_hours_per_year        # 8400 h/yr
    lf_yrs  = _econ.plant_life_years                # 20 yr

    co2_sep = [
        v(fs.flash.F * (1 - fs.flash.psi) * fs.flash.x["CO2"])
        for fs in (fs1, fs2, fs3, fs4, fs5)  
    ]                                               # mol/s
    annual_co2_t = [
        c * MW_CO2 * op_hrs * 3600 / 1000
        for c in co2_sep
    ]                                               # t CO2/yr
    tac          = [c / lf_yrs + o for c, o in zip(capex, opex)]             # €/yr
    opex_per_t   = [o / t for o, t in zip(opex, annual_co2_t)]              # €/t CO2
    lcc_per_t    = [l / (t * lf_yrs) for l, t in zip(lcc, annual_co2_t)]   # €/t CO2

    row("Total compression power [kW]",  comp_power)
    row("HX1 (water) duty [kW_th]",      hx1_duty)
    row("Intercooler(s) duty [kW_th]",   ic_duty)
    row("HX2 (NH3) duty [kW_th]",        hx2_duty)
    row("Total CAPEX [€]",               capex,        fmt="{:>11,.0f}")
    row("Total annual OPEX [€/yr]",      opex,         fmt="{:>11,.0f}")
    row("Total annualized cost [€/yr]",  tac,          fmt="{:>11,.0f}")
    row("Lifecycle cost [€]",            lcc,          fmt="{:>11,.0f}")
    row("Vapour fraction ψ [-]",         psi,          fmt="{:>11.4f}")
    row("CO2 captured [t/yr]",           annual_co2_t, fmt="{:>11,.1f}")
    row("OPEX cost per t CO2 [€/t]",     opex_per_t,   fmt="{:>11.2f}")
    row("LCC per t CO2 [€/t]",           lcc_per_t,    fmt="{:>11.2f}")
    print("=" * W + "\n")