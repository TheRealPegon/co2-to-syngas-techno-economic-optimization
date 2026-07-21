import sys
import os
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from configuration import Economics
from SubModels.PSAModel import PSA

# p_H values to sweep (kPa)
P_H_VALUES = [110, 150, 200, 300, 500]  # up to 5 bar

_op_hours_per_year = 8400
_plant_life = 20

results = []

for p_H_fix in P_H_VALUES:
    print(f"\n{'='*60}")
    print(f"  Solving with p_H fixed = {p_H_fix} kPa ({p_H_fix/100:.1f} bar)")
    print(f"{'='*60}")

    model = PSA(allow_vacuum=True)  # no post-compression (p_product_out left at p_H)
    # model.nf_CO2_psaH_in.fix(252)
    model.nf_CO2_psaH_in.fix(28.38*2)
    model.nf_CO_psaH_in.fix(28.38)
    model.p_H.fix(p_H_fix)

    model.test_obj = pyo.Objective(
        expr=model.CAPEX + _plant_life * _op_hours_per_year * model.op_cost_psa,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=False, options={"max_iter": 5000})

    status = str(result.solver.status)
    term_cond = str(result.solver.termination_condition)

    # Check constraint violations
    max_viol = 0.0
    for c in model.component_objects(pyo.Constraint, active=True):
        for idx in c:
            con = c[idx] if idx is not None else c
            try:
                lb = pyo.value(con.lower) if con.has_lb() else None
                ub = pyo.value(con.upper) if con.has_ub() else None
                body = pyo.value(con.body)
                if lb is not None and body < lb:
                    max_viol = max(max_viol, lb - body)
                if ub is not None and body > ub:
                    max_viol = max(max_viol, body - ub)
            except Exception:
                pass

    try:
        lifecycle = pyo.value(model.CAPEX + _plant_life * _op_hours_per_year * model.op_cost_psa)
        capex = pyo.value(model.CAPEX)
        opex_yr = pyo.value(model.op_cost_psa) * _op_hours_per_year
        m_ads = pyo.value(model.m_ads)
        p_L = pyo.value(model.p_L)
        power_comp = pyo.value(model.power_comp_in)
        power_pump = pyo.value(model.power_pump_in)
        power_postcomp = pyo.value(model.power_postcomp_in) if hasattr(model, "power_postcomp_in") else 0.0
        eta_CO2_H = pyo.value(model.q_CO2_H) / pyo.value(model.q_CO2_s)
        eta_CO2_L = pyo.value(model.q_CO2_L) / pyo.value(model.q_CO2_s)
        working_cap = eta_CO2_H - eta_CO2_L  # normalised [-]
        working_cap_abs = pyo.value(model.q_CO2_H) - pyo.value(model.q_CO2_L)  # mol/kg
        n_trains = pyo.value(model.N_trains)
        co_recovery = pyo.value(model.nf_CO_product) / pyo.value(model.nf_CO_psaH_in)
        saturation_ads_CO2 = pyo.value(model.saturation_ads_CO2)
        saturation_ads_CO = pyo.value(model.saturation_ads_CO)
        saturation_des_CO2 = pyo.value(model.saturation_des_CO2)
        saturation_des_CO = pyo.value(model.saturation_des_CO)
        t_ads = pyo.value(model.t_ads)
        t_des = pyo.value(model.t_des)
        q_CO2_H = pyo.value(model.q_CO2_H)
        q_CO2_L = pyo.value(model.q_CO2_L)
        q_CO_H = pyo.value(model.q_CO_H)
        q_CO_L = pyo.value(model.q_CO_L)

    except Exception:
        lifecycle = capex = opex_yr = m_ads = p_L = power_comp = power_pump = power_postcomp = None
        eta_CO2_H = eta_CO2_L = working_cap = working_cap_abs = n_trains = co_recovery = None
        saturation_ads_CO2 = saturation_ads_CO = saturation_des_CO2 = saturation_des_CO = None
        t_ads = t_des = None
        q_CO2_H = q_CO2_L = q_CO_H = q_CO_L = None

    row = dict(
        p_H=p_H_fix,
        status=status,
        termination=term_cond,
        max_violation=max_viol,
        lifecycle=lifecycle,
        capex=capex,
        opex_yr=opex_yr,
        m_ads=m_ads,
        p_L=p_L,
        power_comp=power_comp,
        power_pump=power_pump,
        power_postcomp=power_postcomp,
        eta_CO2_H=eta_CO2_H,
        eta_CO2_L=eta_CO2_L,
        working_cap=working_cap,
        working_cap_abs=working_cap_abs,
        n_trains=n_trains,
        co_recovery=co_recovery,
        saturation_ads_CO2=saturation_ads_CO2,
        saturation_ads_CO=saturation_ads_CO,
        saturation_des_CO2=saturation_des_CO2,
        saturation_des_CO=saturation_des_CO,
        t_ads=t_ads,
        t_des=t_des,
        q_CO2_H=q_CO2_H,
        q_CO2_L=q_CO2_L,
        q_CO_H=q_CO_H,
        q_CO_L=q_CO_L,
    )
    results.append(row)

    print(f"  Status: {status} | Termination: {term_cond} | Max violation: {max_viol:.3g}")
    if lifecycle is not None:
        print(f"  Lifecycle: €{lifecycle:,.0f} | CAPEX: €{capex:,.0f} | OPEX/yr: €{opex_yr:,.0f}")
        print(
            f"  m_ads: {m_ads:,.0f} kg | p_L: {p_L:.1f} kPa | N_trains: {n_trains:.2f} | P_comp: {power_comp:.1f} kW | P_pump: {power_pump:.1f} kW | P_postcomp: {power_postcomp:.1f} kW"
        )
        print(
            f"  eta_CO2_H: {eta_CO2_H:.3f} | eta_CO2_L: {eta_CO2_L:.3f} | working_cap: {working_cap:.3f} | CO recovery: {co_recovery:.3f}"
        )
        print(
            f"  t_ads: {t_ads:.1f}s | t_des: {t_des:.1f}s | sat_ads CO2/CO: {saturation_ads_CO2:.3f}/{saturation_ads_CO:.3f} | sat_des CO2/CO: {saturation_des_CO2:.3f}/{saturation_des_CO:.3f}"
        )

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'='*140}")
print(
    f"{'p_H (kPa)':>10} {'p_H (bar)':>9} {'Status':>8} {'Termination':>15} {'MaxViol':>9} {'Lifecycle €':>13} {'CAPEX €':>11} {'m_ads kg':>10} {'WorkCap(-)':>11} {'WorkCap(mol/kg)':>16} {'CO Rec.':>9}"
)
print(f"{'='*140}")
for r in results:
    lc = f"{r['lifecycle']:>13,.0f}" if r["lifecycle"] else f"{'—':>13}"
    cap = f"{r['capex']:>11,.0f}" if r["capex"] else f"{'—':>11}"
    ma = f"{r['m_ads']:>10,.0f}" if r["m_ads"] else f"{'—':>10}"
    wc = f"{r['working_cap']:>11.3f}" if r["working_cap"] is not None else f"{'—':>11}"
    wca = f"{r['working_cap_abs']:>16.3f}" if r["working_cap_abs"] is not None else f"{'—':>16}"
    cor = f"{r['co_recovery']:>9.3f}" if r["co_recovery"] is not None else f"{'—':>9}"
    print(
        f"{r['p_H']:>10} {r['p_H']/100:>9.1f} {r['status']:>8} {r['termination']:>15} {r['max_violation']:>9.3g} {lc} {cap} {ma} {wc} {wca} {cor}"
    )

# ── Plots ──────────────────────────────────────────────────────────────────────
feasible = [r for r in results if r["lifecycle"] is not None and r["max_violation"] < 1e-3]

if feasible:
    pH_f = [r["p_H"] / 100 for r in feasible]  # bar
    pL_f = [r["p_L"] / 100 for r in feasible]  # bar
    lc_f = [r["lifecycle"] / 1e6 for r in feasible]
    cap_f = [r["capex"] / 1e6 for r in feasible]
    op_f = [r["opex_yr"] / 1e6 for r in feasible]
    ma_f = [r["m_ads"] / 1e3 for r in feasible]
    pc_f = [r["power_comp"] for r in feasible]
    pp_f = [r["power_pump"] for r in feasible]
    ppc_f = [r["power_postcomp"] for r in feasible]
    wc_f = [r["working_cap"] for r in feasible]

    wca_f = [r["working_cap_abs"] for r in feasible]  # mol/kg
    cor_f = [r["co_recovery"] * 100 for r in feasible]  # %
    sat_ads_CO2_f = [r["saturation_ads_CO2"] * 100 for r in feasible]  # %
    sat_ads_CO_f = [r["saturation_ads_CO"] * 100 for r in feasible]  # %
    sat_des_CO2_f = [r["saturation_des_CO2"] * 100 for r in feasible]  # %
    sat_des_CO_f = [r["saturation_des_CO"] * 100 for r in feasible]  # %
    qCO2H_f = [r["q_CO2_H"] for r in feasible]  # mol/kg
    qCO2L_f = [r["q_CO2_L"] for r in feasible]  # mol/kg
    qCOH_f = [r["q_CO_H"] for r in feasible]  # mol/kg
    qCOL_f = [r["q_CO_L"] for r in feasible]  # mol/kg
    tads_f = [r["t_ads"] for r in feasible]  # s
    tdes_f = [r["t_des"] for r in feasible]  # s
    tcyc_f = [r["t_ads"] + r["t_des"] for r in feasible]  # s (adsorption + desorption)

    # Publication-ready styling applied to every figure produced below.
    plt.rcParams.update({
        "font.size":         12,
        "font.family":       "serif",
        "mathtext.fontset":  "cm",
        "axes.titlesize":    13,
        "axes.titleweight":  "normal",
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "axes.linewidth":    1.0,
        "lines.linewidth":   2.0,
        "lines.markersize":  7,
        "grid.alpha":        0.3,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })

    # Shared axis label: adsorption (high) pressure, spelled out with a math subscript.
    PH_LABEL = r"Adsorption Pressure $p_{\mathrm{H}}$ (bar)"

    # Each panel is defined once as (filename_stem, draw_function). The draw
    # function takes an Axes and renders the panel onto it, so the exact same
    # code feeds both the combined grid and the individual figures.
    def _draw_cost(ax):
        ax.plot(pH_f, lc_f, "o-", color="steelblue", label="Lifecycle (20 yr)")
        ax.plot(pH_f, cap_f, "s--", color="tomato", label="CAPEX")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("Cost (M€)")
        ax.set_title(r"Lifecycle & CAPEX vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_m_ads(ax):
        ax.plot(pH_f, ma_f, "o-", color="darkorange")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("Adsorbent Mass (t)")
        ax.set_title(r"Adsorbent Mass vs Adsorption Pressure")
        ax.grid(True)

    def _draw_power(ax):
        ax.plot(pH_f, pc_f, "o-", color="royalblue", label="Compressor")
        ax.plot(pH_f, pp_f, "s--", color="mediumseagreen", label="Vacuum pump")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("Shaft Power (kW)")
        ax.set_title(r"Power Consumption vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_p_L(ax):
        ax.plot(pH_f, pL_f, "o-", color="crimson")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"Desorption Pressure $p_{\mathrm{L}}$ (bar)")
        ax.set_title(r"Desorption Pressure vs Adsorption Pressure")
        ax.grid(True)

    def _draw_wca(ax):
        ax.plot(pH_f, wca_f, "o-", color="teal")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"$\Delta q_{\mathrm{CO_2}}$ (mol kg$^{-1}$)")
        ax.set_title(r"Absolute Working Capacity vs Adsorption Pressure")
        ax.grid(True)

    def _draw_wc(ax):
        ax.plot(pH_f, wc_f, "s--", color="purple")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"$(q_{\mathrm{CO_2},H} - q_{\mathrm{CO_2},L})\,/\,q_{\mathrm{CO_2,s}}$ (–)")
        ax.set_title(r"Normalised Working Capacity vs Adsorption Pressure")
        ax.grid(True)

    def _draw_co_recovery(ax):
        ax.plot(pH_f, cor_f, "o-", color="darkgreen")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("CO Recovery (%)")
        ax.set_title(r"CO Recovery vs Adsorption Pressure")
        ax.grid(True)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    def _draw_qCO2(ax):
        ax.plot(pH_f, qCO2H_f, "o-", color="steelblue", label=r"$q_{\mathrm{CO_2}}$ at $p_{\mathrm{H}}$ (ads.)")
        ax.plot(pH_f, qCO2L_f, "s--", color="lightskyblue", label=r"$q_{\mathrm{CO_2}}$ at $p_{\mathrm{L}}$ (des.)")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"$q_{\mathrm{CO_2}}$ (mol kg$^{-1}$)")
        ax.set_title(r"CO$_2$ Equilibrium Loading vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_qCO(ax):
        ax.plot(pH_f, qCOH_f, "o-", color="tomato", label=r"$q_{\mathrm{CO}}$ at $p_{\mathrm{H}}$ (ads.)")
        ax.plot(pH_f, qCOL_f, "s--", color="lightsalmon", label=r"$q_{\mathrm{CO}}$ at $p_{\mathrm{L}}$ (des.)")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"$q_{\mathrm{CO}}$ (mol kg$^{-1}$)")
        ax.set_title(r"CO Equilibrium Loading vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_q_combined(ax):
        ax.plot(pH_f, qCO2H_f, "o-", color="steelblue", label=r"$q_{\mathrm{CO_2}}$ at $p_{\mathrm{H}}$ (ads.)")
        ax.plot(pH_f, qCO2L_f, "s--", color="lightskyblue", label=r"$q_{\mathrm{CO_2}}$ at $p_{\mathrm{L}}$ (des.)")
        ax.plot(pH_f, qCOH_f, "o-", color="tomato", label=r"$q_{\mathrm{CO}}$ at $p_{\mathrm{H}}$ (ads.)")
        ax.plot(pH_f, qCOL_f, "s--", color="lightsalmon", label=r"$q_{\mathrm{CO}}$ at $p_{\mathrm{L}}$ (des.)")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel(r"Equilibrium Loading $q$ (mol kg$^{-1}$)")
        ax.set_title("CO$_2$ & CO Equilibrium Loading vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_cycle_time(ax):
        ax.plot(pH_f, tads_f, "o-", color="darkgoldenrod", label="Adsorption $t_{\\mathrm{ads}}$")
        ax.plot(pH_f, tdes_f, "s--", color="sienna", label="Desorption $t_{\\mathrm{des}}$")
        ax.plot(pH_f, tcyc_f, "^-", color="black", label="Total cycle $t_{\\mathrm{ads}}+t_{\\mathrm{des}}$")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("Time (s)")
        ax.set_title(r"Cycle Time vs Adsorption Pressure")
        ax.legend()
        ax.grid(True)

    def _draw_saturation(ax):
        ax.plot(pH_f, sat_ads_CO2_f, "o-", color="indigo", label=r"CO$_2$ adsorption")
        ax.plot(pH_f, sat_ads_CO_f, "o--", color="steelblue", label=r"CO adsorption")
        ax.plot(pH_f, sat_des_CO2_f, "s-", color="mediumorchid", label=r"CO$_2$ desorption")
        ax.plot(pH_f, sat_des_CO_f, "s--", color="lightsalmon", label=r"CO desorption")
        ax.set_xlabel(PH_LABEL)
        ax.set_ylabel("Bed Saturation (%)")
        ax.set_title(r"Per-Species Bed Saturation vs Adsorption Pressure")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    panels = [
        ("cost_capex",        _draw_cost),
        ("adsorbent_mass",    _draw_m_ads),
        ("power",             _draw_power),
        ("desorption_pressure", _draw_p_L),
        ("working_cap_abs",   _draw_wca),
        ("working_cap_norm",  _draw_wc),
        ("co_recovery",       _draw_co_recovery),
        ("qCO2_loading",      _draw_qCO2),
        ("qCO_loading",       _draw_qCO),
        ("q_combined_loading", _draw_q_combined),
        ("cycle_time",        _draw_cycle_time),
        ("bed_saturation",    _draw_saturation),
    ]

    # Plots folder one level up from Scripts/ (i.e. PSA/Plots/).
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Combined grid: 2 columns, enough rows for all panels (sized from the count).
    ncols = 2
    nrows = -(-len(panels) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    fig.suptitle(r"PSA parametric sweep — $p_{\mathrm{H}}$ fixed, vacuum desorption, no post-compression",
                 fontsize=14)
    for ax, (_, draw) in zip(axes.flat, panels):
        draw(ax)
    # Hide any unused axes (when len(panels) is odd).
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
    combined_out = os.path.join(plots_dir, "sweep_pH.png")
    plt.savefig(combined_out, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(combined_out)}")

    # Individual figures: one PNG per panel in a sub-folder of the Plots folder.
    indiv_dir = os.path.join(plots_dir, "sweep_pH_individual")
    os.makedirs(indiv_dir, exist_ok=True)
    for stem, draw in panels:
        f_i, ax_i = plt.subplots(figsize=(7, 5))
        draw(ax_i)
        f_i.tight_layout()
        out_i = os.path.join(indiv_dir, f"sweep_pH_{stem}.png")
        f_i.savefig(out_i, dpi=150)
        plt.close(f_i)
    print(f"Individual plots saved to {os.path.abspath(indiv_dir)} ({len(panels)} files)")

    # Save the plotted data as a CSV in the Data/ folder one level up (PSA/Data/).
    import csv
    data_dir = os.path.join(os.path.dirname(__file__), "..", "Data")
    os.makedirs(data_dir, exist_ok=True)
    # (column header, per-point values) — one column per plotted series.
    csv_columns = [
        ("p_H (bar)",                 pH_f),
        ("lifecycle (M€)",            lc_f),
        ("CAPEX (M€)",                cap_f),
        ("OPEX (M€/yr)",              op_f),
        ("adsorbent_mass (t)",        ma_f),
        ("power_comp (kW)",           pc_f),
        ("power_pump (kW)",           pp_f),
        ("power_postcomp (kW)",       ppc_f),
        ("p_L (bar)",                 pL_f),
        ("working_cap_abs (mol/kg)",  wca_f),
        ("working_cap_norm (-)",      wc_f),
        ("CO_recovery (%)",           cor_f),
        ("q_CO2_H (mol/kg)",          qCO2H_f),
        ("q_CO2_L (mol/kg)",          qCO2L_f),
        ("q_CO_H (mol/kg)",           qCOH_f),
        ("q_CO_L (mol/kg)",           qCOL_f),
        ("t_ads (s)",                 tads_f),
        ("t_des (s)",                 tdes_f),
        ("t_cycle (s)",               tcyc_f),
        ("sat_ads_CO2 (%)",           sat_ads_CO2_f),
        ("sat_ads_CO (%)",            sat_ads_CO_f),
        ("sat_des_CO2 (%)",           sat_des_CO2_f),
        ("sat_des_CO (%)",            sat_des_CO_f),
    ]
    out_csv = os.path.join(data_dir, "sweep_pH.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([h for h, _ in csv_columns])
        for row in zip(*[vals for _, vals in csv_columns]):
            writer.writerow(row)
    print(f"Data saved to {os.path.abspath(out_csv)}")

    plt.show()
else:
    print("\nNo feasible points to plot.")
