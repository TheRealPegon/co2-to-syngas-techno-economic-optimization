"""
Unit test: AbsorberSurrogate simulation sweep vs recorded CSV data.

Validates individual physical metrics first (H_m, D_m, T_liq_bot,
Q_intercooler, A_intercooler, delta_P), then CAPEX/OPEX derived quantities.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SubModels")))

from SubModels.AbsorberSurrogateModel import AbsorberSurrogate

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../SurrogateModeling/AmineWashing/Absorber/Data/absorber_scalar.csv",
)

# ── Acceptance tolerances ──────────────────────────────────────────────────────
TOL = {
    "H_m":               0.05,   # 5%  — power law R²=0.9984
    "D_m":               0.05,   # 5%
    "T_liq_bot_C":       0.01,   # 1%  — exponential R²=0.9956  (range ~44-48°C)
    "Q_intercooler_W":   0.15,   # 15% — 2D power law R²=0.9755 (weakest fit)
    "A_intercooler_m2":  0.05,   # 5%  — 2D power law R²=0.9986
    "delta_P_Pa":        0.05,   # 5%  — power law R²=0.9979
    "CAPEX_EUR_2024":    0.30,   # 30% — derived, methodology may differ
    "OPEX_cw_EUR_yr":    0.30,   # 30% — derived from Q_ic
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[(df["T_lean_C"] != 55.0) & (df["status"] == "locallyOptimal")]
    return df.sort_values(["T_lean_C", "F_co2_mol_s"]).reset_index(drop=True)


def run_model(row):
    model = AbsorberSurrogate(
        F_co2_mol_s=row["F_co2_mol_s"],
        T_lean_C=row["T_lean_C"],
        F_gas_total_mol_s=row["V_mol_s"],
    )
    return model.evaluate()


df_global = load_data()
test_cases = [
    (int(row["F_co2_mol_s"]), int(row["T_lean_C"]), row)
    for _, row in df_global.iterrows()
]
ids = [f"F{t[0]}_T{t[1]}" for t in test_cases]


# ── Physical metric tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_column_height(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["H_m"] - row["H_m"]) / row["H_m"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['H_m']:.3f} m | rec={row['H_m']:.3f} m | err={err:.1%}")
    assert err <= TOL["H_m"], f"H_m error {err:.1%} > {TOL['H_m']:.0%}"


@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_column_diameter(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["D_m"] - row["D_m"]) / row["D_m"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['D_m']:.3f} m | rec={row['D_m']:.3f} m | err={err:.1%}")
    assert err <= TOL["D_m"], f"D_m error {err:.1%} > {TOL['D_m']:.0%}"


@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_liquid_bottom_temperature(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["T_liq_bot_C"] - row["T_liq_bot_C"]) / row["T_liq_bot_C"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['T_liq_bot_C']:.3f} °C | rec={row['T_liq_bot_C']:.3f} °C | err={err:.1%}")
    assert err <= TOL["T_liq_bot_C"], f"T_liq_bot error {err:.1%} > {TOL['T_liq_bot_C']:.0%}"


@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_intercooler_duty(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["Q_intercooler_W"] - row["Q_intercooler_W"]) / row["Q_intercooler_W"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['Q_intercooler_W']/1e6:.3f} MW "
          f"| rec={row['Q_intercooler_W']/1e6:.3f} MW | err={err:.1%}")
    assert err <= TOL["Q_intercooler_W"], f"Q_intercooler error {err:.1%} > {TOL['Q_intercooler_W']:.0%}"


@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_intercooler_area(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["A_intercooler_m2"] - row["A_intercooler_m2"]) / row["A_intercooler_m2"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['A_intercooler_m2']:.1f} m² "
          f"| rec={row['A_intercooler_m2']:.1f} m² | err={err:.1%}")
    assert err <= TOL["A_intercooler_m2"], f"A_intercooler error {err:.1%} > {TOL['A_intercooler_m2']:.0%}"


@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_pressure_drop(F_co2, T_lean, row):
    res = run_model(row)
    err = abs(res["delta_P_Pa"] - row["delta_P_Pa"]) / row["delta_P_Pa"]
    print(f"\n  F={F_co2}, T={T_lean} | surr={res['delta_P_Pa']:.1f} Pa "
          f"| rec={row['delta_P_Pa']:.1f} Pa | err={err:.1%}")
    assert err <= TOL["delta_P_Pa"], f"delta_P error {err:.1%} > {TOL['delta_P_Pa']:.0%}"


# ── CAPEX / OPEX tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("F_co2,T_lean,row", test_cases, ids=ids)
def test_cw_opex(F_co2, T_lean, row):
    res = run_model(row)
    surr = res["OPEX_cw_EUR_yr"] / 0.96   # EUR→USD for comparison
    rec  = row["intercooler_cw_oc_USD_yr"]
    err  = abs(surr - rec) / rec
    print(f"\n  F={F_co2}, T={T_lean} | surr={surr:,.0f} USD/yr | rec={rec:,.0f} USD/yr | err={err:.1%}")
    assert err <= TOL["OPEX_cw_EUR_yr"], f"CW OPEX error {err:.1%} > {TOL['OPEX_cw_EUR_yr']:.0%}"


# ── Standalone sweep summary ───────────────────────────────────────────────────

METRICS = [
    ("H_m",              "H_m",             "m",     1,    "H_m"),
    ("D_m",              "D_m",             "m",     1,    "D_m"),
    ("T_liq_bot_C",      "T_liq_bot_C",     "°C",    1,    "T_liq_bot_C"),
    ("Q_intercooler_W",  "Q_intercooler_W", "MW",    1e-6, "Q_intercooler_W"),
    ("A_intercooler_m2", "A_intercooler_m2","m²",    1,    "A_intercooler_m2"),
    ("delta_P_Pa",       "delta_P_Pa",      "Pa",    1,    "delta_P_Pa"),
]


def print_sweep_summary():
    df = load_data()

    for surr_key, csv_col, unit, scale, tol_key in METRICS:
        tol = TOL[tol_key]
        print(f"\n{'─'*80}")
        print(f"  {surr_key}   (tolerance: {tol:.0%})")
        print(f"  {'F_co2':>6} {'T_lean':>7} {'Surrogate':>12} {'Recorded':>12} {'Err%':>8}  {'':>2}")
        print(f"  {'':-<6} {'':-<7} {'':-<12} {'':-<12} {'':-<8}")
        errs = []
        for _, row in df.iterrows():
            res  = run_model(row)
            surr = res[surr_key] * scale
            rec  = row[csv_col]  * scale
            err  = (surr - rec) / rec * 100
            errs.append(abs(err))
            flag = " !" if abs(err) > tol * 100 else ""
            print(f"  {row['F_co2_mol_s']:>6.0f} {row['T_lean_C']:>7.0f} "
                  f"{surr:>12.3f} {rec:>12.3f} {err:>+7.1f}%{flag}")
        print(f"  {'Mean abs err':>28}: {sum(errs)/len(errs):.1f}%  "
              f"Max: {max(errs):.1f}%  [{unit}]")

    # CW OPEX
    print(f"\n{'─'*80}")
    print(f"  intercooler_cw_oc_USD_yr   (tolerance: {TOL['OPEX_cw_EUR_yr']:.0%})")
    print(f"  {'F_co2':>6} {'T_lean':>7} {'Surrogate':>14} {'Recorded':>14} {'Err%':>8}")
    print(f"  {'':-<6} {'':-<7} {'':-<14} {'':-<14} {'':-<8}")
    errs = []
    for _, row in df.iterrows():
        res  = run_model(row)
        surr = res["OPEX_cw_EUR_yr"] / 0.96
        rec  = row["intercooler_cw_oc_USD_yr"]
        err  = (surr - rec) / rec * 100
        errs.append(abs(err))
        flag = " !" if abs(err) > TOL["OPEX_cw_EUR_yr"] * 100 else ""
        print(f"  {row['F_co2_mol_s']:>6.0f} {row['T_lean_C']:>7.0f} "
              f"{surr:>14,.0f} {rec:>14,.0f} {err:>+7.1f}%{flag}")
    print(f"  {'Mean abs err':>28}: {sum(errs)/len(errs):.1f}%  Max: {max(errs):.1f}%  [USD/yr]")


if __name__ == "__main__":
    print_sweep_summary()
