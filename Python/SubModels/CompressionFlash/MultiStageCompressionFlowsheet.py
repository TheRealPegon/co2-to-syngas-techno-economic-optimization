import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pyomo.environ as pyo

from configuration import Economics
from SubModels.CompressionFlash.Compressor import MolarCompressorModel
from SubModels.CompressionFlash.HeatEXModel import HeatExchangerModel
from SubModels.CompressionFlash.FlashModel import FlashModel
from SubModels.CompressionFlash.MixtureCp import MixtureCpModel


class MultiStageCompressionFlowsheet(pyo.ConcreteModel):
    """
    N-stage compression with intercooling and optional flash separation.

    Topology:
      Comp1 -> [IC1 -> Comp2 -> IC2 -> ... -> Comp(N-1) -> IC(N-1) ->] CompN
            -> HX1 (water, T_inter) -> HX2 (NH3, T_final) [-> Flash]

    For n_stages=1 there are no intercoolers.
    For with_flash=True, K_dict must be supplied; the flash drum sits after HX2.

    Intermediate pressures P_inter[k] (k=1..n-1) and intercooler outlet
    temperatures T_ic_out[k] are free optimisation variables; the objective
    minimises total lifecycle cost (CAPEX + plant_life * OPEX).

    Enthalpy duties use the Shomate ideal-gas integral (MixtureCpModel).
    When with_flash=True, HX2 duty includes a CO2 condensation latent-heat
    correction (ΔHvap ≈ 11.5 kJ/mol at 30 bar / 270 K, NIST).
    """

    def __init__(
        self,
        z_dict,              # {component: mole_fraction}
        feed_flow,           # mol/s
        n_stages=1,          # number of compression stages
        with_flash=False,             # include flash drum after HX2
        K_dict=None,                  # {component: K_value} — required if with_flash=True
        Pin=1.0,                      # bar
        Tin=298.15,                   # K  (25 °C)
        Pout=30.0,                    # bar — initial guess / fixed value
        float_outlet_pressure=False,  # make Pout a free optimisation variable [1, 50] bar
        T_inter=303.15,      # K  (30 °C)  — HX1 outlet / HX2 inlet
        T_final=238.15,      # K  (-35 °C) — HX2 outlet (flash temp if with_flash)
        gamma=1.3,
        eta=0.9,
        name=None,
    ):
        if name is None:
            flash_str = "Flash" if with_flash else ""
            name = f"{n_stages}StageCompression{flash_str}Flowsheet"
        super().__init__(name=name)

        if with_flash and K_dict is None:
            raise ValueError("K_dict must be provided when with_flash=True")

        n = n_stages

        # ------------------------------------------------------------------
        # Intermediate pressure and intercooler outlet temperature variables
        # (only present for n > 1: one per intercooler stage k = 1..n-1)
        # ------------------------------------------------------------------
        # Equal-ratio initial guesses: P_inter[k] = Pin * (Pout/Pin)^(k/n)
        ratio = (Pout / Pin) ** (1.0 / n) if n > 1 else 1.0
        P_inits = [Pin * ratio ** k for k in range(1, n)]  # length n-1

        self._P_inter = []   # Python list of Pyomo Vars (index 0 = stage 1)
        self._T_ic_out = []  # Python list of Pyomo Vars

        _p_upper = 50.0 * 0.99 if float_outlet_pressure else Pout * 0.99
        for k in range(1, n):
            p_var = pyo.Var(initialize=P_inits[k - 1], bounds=(Pin * 1.01, _p_upper))
            t_var = pyo.Var(initialize=303.15, bounds=(303.15, 500.0))
            setattr(self, f"P_inter{k}", p_var)
            setattr(self, f"T_ic_out{k}", t_var)
            self._P_inter.append(p_var)
            self._T_ic_out.append(t_var)

        # Ordering constraints: P_inter[k] <= P_inter[k+1]
        for k in range(len(self._P_inter) - 1):
            setattr(self, f"eq_pressure_order{k+1}{k+2}",
                    pyo.Constraint(expr=self._P_inter[k] <= self._P_inter[k + 1]))

        # When outlet pressure floats: create Pout_var and ensure last P_inter <= Pout_var
        if float_outlet_pressure:
            self.Pout_var = pyo.Var(initialize=Pout, bounds=(2.0, 50.0))
            if self._P_inter:
                self.eq_pressure_order_final = pyo.Constraint(
                    expr=self._P_inter[-1] <= self.Pout_var
                )

        # ------------------------------------------------------------------
        # MixtureCp models
        #   cp_ic[k]  (k=1..n-1): one per intercooler
        #     T_hot  → comp[k].Tout (linked below)
        #     T_cold → T_ic_out[k]  (linked below)
        #   cp_hx: final HX1/HX2
        #     T_hot  → comp[n].Tout (linked below)
        #     T_cold  = T_inter
        #     T_cold2 = T_final
        # ------------------------------------------------------------------
        self._cp_ic = []
        for k in range(1, n):
            cp = MixtureCpModel(z_dict=z_dict, T_init=Tin, name=f"MixtureCp_IC{k}")
            cp.T.fix(Tin)
            cp.T_cold2.fix(Tin)  # unused slot
            cp.obj.deactivate()
            setattr(self, f"cp_ic{k}", cp)
            self._cp_ic.append(cp)

        self.cp_hx = MixtureCpModel(z_dict=z_dict, T_init=Tin, name="MixtureCp_HX")
        self.cp_hx.T.fix(Tin)
        self.cp_hx.T_cold.fix(T_inter)
        self.cp_hx.T_cold2.fix(T_final if with_flash else T_inter)
        self.cp_hx.obj.deactivate()

        # ------------------------------------------------------------------
        # Compressors: comp[k] for k = 1..n
        #   Stage 1 : Pin → P_inter[1]   (or Pout if n=1), inlet = Tin
        #   Stage k : P_inter[k-1] → P_inter[k] (or Pout), inlet = T_ic_out[k-1]
        # ------------------------------------------------------------------
        self._comps = []
        for k in range(1, n + 1):
            Pin_k  = Pin        if k == 1 else P_inits[k - 2]
            Pout_k = Pout       if k == n else P_inits[k - 1]
            comp_name = "MolarCompressor" if k == 1 else f"MolarCompressor{k}"
            comp = MolarCompressorModel(
                Tin=Tin, Pin=Pin_k, Pout=Pout_k,
                gamma=gamma, cp_molar=1.0,
                molar_flow=feed_flow, eta=eta,
                name=comp_name,
            )
            comp.obj.deactivate()
            setattr(self, f"comp{k}", comp)
            self._comps.append(comp)

        # ------------------------------------------------------------------
        # Intercoolers: ic[k] for k = 1..n-1
        # ------------------------------------------------------------------
        self._ics = []
        for k in range(1, n):
            ic = HeatExchangerModel(
                Tin=0.0, Tout=303.15,
                molar_flow=feed_flow, cp_molar=1.0,
                cooling_medium="water", name=f"Intercooler{k}",
            )
            ic.obj.deactivate()
            setattr(self, f"ic{k}", ic)
            self._ics.append(ic)

        # ------------------------------------------------------------------
        # Final coolers: HX1 (water) and HX2 (NH3)
        # ------------------------------------------------------------------
        self.hx1 = HeatExchangerModel(
            Tin=0.0, Tout=T_inter,
            molar_flow=feed_flow, cp_molar=1.0,
            cooling_medium="water",
        )
        self.hx1.obj.deactivate()

        # ------------------------------------------------------------------
        # Optional flash drum + HX2 (NH3 refrigeration to T_final)
        # Only created when with_flash=True; otherwise HX1 alone cools to T_inter.
        # ------------------------------------------------------------------
        if with_flash:
            self.hx2 = HeatExchangerModel(
                Tin=T_inter, Tout=T_final,
                molar_flow=feed_flow, cp_molar=1.0,
                cooling_medium="NH3", name="HeatExchanger_NH3",
            )
            self.hx2.obj.deactivate()
            self.flash = FlashModel(z_dict, K_dict, feed_flow, Pout, T_final)

        # ------------------------------------------------------------------
        # Compressor temperature and power constraints
        #
        # For each stage k:
        #   P_in_expr  = comp.Pin (fixed)        if k == 1
        #              = P_inter[k-2] (Var)       otherwise
        #   P_out_expr = comp.Pout (fixed)        if k == n
        #              = P_inter[k-1] (Var)       otherwise
        #   T_in_expr  = comp.Tin (fixed = Tin)   if k == 1
        #              = T_ic_out[k-2] (Var)      otherwise
        #   cp_mix_k   = cp_ic[k-1].cp_mix        if k < n  (intercooler follows)
        #              = cp_hx.cp_mix             if k == n  (HX follows)
        # ------------------------------------------------------------------
        for k in range(1, n + 1):
            comp = self._comps[k - 1]

            P_in_expr  = comp.Pin  if k == 1 else self._P_inter[k - 2]
            if k == n:
                P_out_expr = self.Pout_var if float_outlet_pressure else comp.Pout
            else:
                P_out_expr = self._P_inter[k - 1]
            T_in_expr  = comp.Tin  if k == 1 else self._T_ic_out[k - 2]
            cp_mix_k   = self.cp_hx.cp_mix if k == n else self._cp_ic[k - 1].cp_mix

            comp.eq_temperature.deactivate()
            setattr(self, f"eq_comp{k}_temp", pyo.Constraint(
                expr=comp.Tout
                == T_in_expr
                + (T_in_expr / comp.eta)
                * ((P_out_expr / P_in_expr) ** ((comp.gamma - 1) / comp.gamma) - 1)
            ))

            comp.eq_power.deactivate()
            setattr(self, f"eq_comp{k}_power", pyo.Constraint(
                expr=comp.power_kw
                == comp.molar_flow * cp_mix_k * (comp.Tout - T_in_expr) / 1000.0
            ))

        # ------------------------------------------------------------------
        # Intercooler cp T_hot/T_cold links and duty constraints
        # ------------------------------------------------------------------
        for k in range(1, n):
            cp  = self._cp_ic[k - 1]
            ic  = self._ics[k - 1]
            T_ic_out = self._T_ic_out[k - 1]

            setattr(self, f"eq_cp_ic{k}_T_hot",
                    pyo.Constraint(expr=cp.T_hot == self._comps[k - 1].Tout))
            setattr(self, f"eq_cp_ic{k}_T_cold",
                    pyo.Constraint(expr=cp.T_cold == T_ic_out))

            ic.eq_duty.deactivate()
            setattr(self, f"eq_ic{k}_duty",
                    pyo.Constraint(expr=ic.duty_kw == ic.molar_flow * cp.delta_h_mix))

        # ------------------------------------------------------------------
        # Link cp_hx.T_hot → final compressor outlet
        # ------------------------------------------------------------------
        self.eq_cp_hx_T_hot = pyo.Constraint(
            expr=self.cp_hx.T_hot == self._comps[-1].Tout
        )

        # ------------------------------------------------------------------
        # HX1 duty: H(comp_n.Tout) - H(T_inter)
        # ------------------------------------------------------------------
        self.hx1.eq_duty.deactivate()
        self.eq_hx1_duty = pyo.Constraint(
            expr=self.hx1.duty_kw == self.hx1.molar_flow * self.cp_hx.delta_h_mix
        )

        # ------------------------------------------------------------------
        # HX2 duty: H(T_inter) - H(T_final)  [only when with_flash=True]
        # Adds CO2 condensation latent heat (ΔHvap ≈ 11.5 kJ/mol,
        # NIST WebBook, CO2 saturation at ~30 bar / 270 K).
        # ------------------------------------------------------------------
        if with_flash:
            self.hx2.eq_duty.deactivate()
            self.delta_H_vap_CO2 = pyo.Param(initialize=11.5)  # kJ/mol
            condensed_CO2 = self.flash.F * (1 - self.flash.psi) * self.flash.x["CO2"]
            self.eq_hx2_duty = pyo.Constraint(
                expr=self.hx2.duty_kw
                == self.hx2.molar_flow * self.cp_hx.delta_h_mix2
                + condensed_CO2 * self.delta_H_vap_CO2
            )

        # ------------------------------------------------------------------
        # Economics aggregation
        # ------------------------------------------------------------------
        self.total_capex = pyo.Var()
        self.total_opex = pyo.Var()

        capex_expr = (
            sum(c.capex_total for c in self._comps)
            + sum(ic.capex_total for ic in self._ics)
            + self.hx1.capex_total
        )
        opex_expr = (
            sum(c.opex_annual for c in self._comps)
            + sum(ic.opex_annual for ic in self._ics)
            + self.hx1.opex_annual
        )
        if with_flash:
            capex_expr = capex_expr + self.hx2.capex_total + self.flash.CAPEX
            opex_expr  = opex_expr  + self.hx2.opex_annual + self.flash.opex_annual

        self.eq_total_capex = pyo.Constraint(expr=self.total_capex == capex_expr)
        self.eq_total_opex  = pyo.Constraint(expr=self.total_opex  == opex_expr)

        plant_life = Economics().plant_life_years
        self.total_lifecycle_cost = pyo.Var()
        self.eq_lifecycle_cost = pyo.Constraint(
            expr=self.total_lifecycle_cost
            == self.total_capex + plant_life * self.total_opex
        )

        self.obj = pyo.Objective(expr=self.total_lifecycle_cost, sense=pyo.minimize)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def solve(self, solver_name="ipopt", show_log=False):
        return pyo.SolverFactory(solver_name).solve(self, tee=show_log)

    def report(self):
        n = len(self._comps)
        with_flash = hasattr(self, "flash")

        print("\n" + "=" * 60)
        print(f"  {n}-STAGE COMPRESSION {'+ FLASH ' if with_flash else ''}FLOWSHEET SUMMARY")
        print("=" * 60)

        print(f"\n  Cp_mix (at {pyo.value(self.cp_hx.T) - 273.15:.1f} °C) : "
              f"{pyo.value(self.cp_hx.cp_mix):.4f} J/mol·K")

        if n > 1:
            _Pout_report = pyo.value(self.Pout_var) if hasattr(self, "Pout_var") else pyo.value(self._comps[-1].Pout)
            P_vals = [pyo.value(p) for p in self._P_inter]
            T_ic_vals = [pyo.value(t) - 273.15 for t in self._T_ic_out]
            ref_pressures = [
                pyo.value(self._comps[0].Pin) * (_Pout_report
                / pyo.value(self._comps[0].Pin)) ** (k / n)
                for k in range(1, n)
            ]
            print(f"\n  Intermediate pressures (bar):")
            for k, (pv, ref) in enumerate(zip(P_vals, ref_pressures), start=1):
                print(f"    P_inter{k} = {pv:.3f}  [equal-ratio ref: {ref:.3f}]")
            print(f"  Intercooler outlet temperatures (°C):")
            for k, tv in enumerate(T_ic_vals, start=1):
                print(f"    T_ic_out{k} = {tv:.2f}")

        for k, comp in enumerate(self._comps, start=1):
            T_in_val = (pyo.value(self._T_ic_out[k - 2]) if k > 1 else pyo.value(comp.Tin))
            T_out_val = pyo.value(comp.Tout)
            print(f"\n--- Compressor {k} ---")
            print(f"  Inlet  T : {T_in_val - 273.15:.2f} °C")
            print(f"  Outlet T : {T_out_val - 273.15:.2f} °C")
            if k == 1:
                Pin_val = pyo.value(comp.Pin)
            else:
                Pin_val = pyo.value(self._P_inter[k - 2])
            if k == n:
                Pout_val = pyo.value(self.Pout_var) if hasattr(self, "Pout_var") else pyo.value(comp.Pout)
            else:
                Pout_val = pyo.value(self._P_inter[k - 1])
            print(f"  Pressure : {Pin_val:.3f} -> {Pout_val:.3f} bar")
            print(f"  Power    : {pyo.value(comp.power_kw):.2f} kW")
            print(f"  CAPEX    : €{pyo.value(comp.capex_total):,.2f}")
            print(f"  OPEX/yr  : €{pyo.value(comp.opex_annual):,.2f}")

            if k < n:
                ic  = self._ics[k - 1]
                cp  = self._cp_ic[k - 1]
                T_ic_out_val = pyo.value(self._T_ic_out[k - 1])
                print(f"--- Intercooler {k} (water) ---")
                print(f"  Outlet T : {T_ic_out_val - 273.15:.2f} °C")
                print(f"  ΔH_mix   : {pyo.value(cp.delta_h_mix):.4f} kJ/mol")
                print(f"  Duty     : {pyo.value(ic.duty_kw):.2f} kW_th")
                print(f"  CAPEX    : €{pyo.value(ic.capex_total):,.2f}")
                print(f"  OPEX/yr  : €{pyo.value(ic.opex_annual):,.2f}")

        print("\n--- HX1 (water cooling) ---")
        print(f"  Inlet  T : {pyo.value(self._comps[-1].Tout) - 273.15:.2f} °C")
        print(f"  Outlet T : {pyo.value(self.hx1.Tout_target) - 273.15:.2f} °C")
        print(f"  ΔH_mix   : {pyo.value(self.cp_hx.delta_h_mix):.4f} kJ/mol")
        print(f"  Duty     : {pyo.value(self.hx1.duty_kw):.2f} kW_th")
        print(f"  CAPEX    : €{pyo.value(self.hx1.capex_total):,.2f}")
        print(f"  OPEX/yr  : €{pyo.value(self.hx1.opex_annual):,.2f}")

        if with_flash:
            print("\n--- HX2 (NH3 refrigeration) ---")
            print(f"  Inlet  T : {pyo.value(self.hx2.Tin) - 273.15:.2f} °C")
            print(f"  Outlet T : {pyo.value(self.hx2.Tout_target) - 273.15:.2f} °C")
            print(f"  ΔH_mix (sensible) : {pyo.value(self.cp_hx.delta_h_mix2):.4f} kJ/mol")
            condensed = pyo.value(
                self.flash.F * (1 - self.flash.psi) * self.flash.x["CO2"]
            )
            print(f"  CO2 condensed     : {condensed:.4f} mol/s  "
                  f"(ΔHvap = {pyo.value(self.delta_H_vap_CO2):.1f} kJ/mol)")
            print(f"  Latent heat term  : {condensed * pyo.value(self.delta_H_vap_CO2):.2f} kW_th")
            print(f"  Duty (total)      : {pyo.value(self.hx2.duty_kw):.2f} kW_th")
            print(f"  CAPEX    : €{pyo.value(self.hx2.capex_total):,.2f}")
            print(f"  OPEX/yr  : €{pyo.value(self.hx2.opex_annual):,.2f}")

        if with_flash:
            print("\n--- Flash Drum ---")
            print(f"  Vapor fraction (ψ) : {pyo.value(self.flash.psi):.4f}")
            print(f"  Vessel volume      : {pyo.value(self.flash.vessel_volume):.3f} m³")
            print(f"  CAPEX              : €{pyo.value(self.flash.CAPEX):,.2f}")
            V = pyo.value(self.flash.F * self.flash.psi)
            L = pyo.value(self.flash.F * (1 - self.flash.psi))
            print(f"  Vapor flow         : {V:.4f} mol/s")
            print(f"  Liquid flow        : {L:.4f} mol/s")
            print(f"  {'Component':<10} | {'x_i (liq)':>12} | {'y_i (vap)':>12}")
            print(f"  {'-' * 40}")
            for i in self.flash.I:
                print(
                    f"  {i:<10} | {pyo.value(self.flash.x[i]):>12.4f}"
                    f" | {pyo.value(self.flash.y[i]):>12.4f}"
                )

        print("\n" + "=" * 60)
        print(f"{'FLOWSHEET TOTALS':^60}")
        print("=" * 60)
        plant_life = Economics().plant_life_years
        print(f"  Total CAPEX                    : €{pyo.value(self.total_capex):,.2f}")
        print(f"  Total Annual OPEX              : €{pyo.value(self.total_opex):,.2f}")
        print(f"  Lifecycle Cost ({plant_life} yr) : €{pyo.value(self.total_lifecycle_cost):,.2f}")
        print("=" * 60 + "\n")
