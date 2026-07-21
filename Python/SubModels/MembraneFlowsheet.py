"""
MembraneFlowsheet
=================
Combines the CO2 electrolyzer surrogate model, N-stage compression flowsheet
(with optional flash drum), and gas permeation membrane into a single Pyomo
optimisation model.

Process topology
----------------

    [CO2 fresh feed]
           |
           v
    +------------------+  <-- CO2/CO recycle (flash liquid + membrane permeate
    |  CO2 Electrolyzer |                    or mem2 retentate when recycle_permeate_H2)
    +--------+---------+
             |  CO2 + CO + H2  (cathode gas, atmospheric pressure)
             v
    +--------------------------------+
    |  N-stage compression           |
    |  + intercooling                |
    |  + HX1  (water,  T_inter)      |
    |  + HX2  (NH3,    T_final)      |
    +--------+-----------------------+
             |  compressed + cooled gas  (Pout bar)
             |
     with_flash=True only:
             v
    +--------------+
    |  Flash Drum  |---> liquid  (CO2-rich)  ---------> recycle
    +------+-------+
           |  vapour (CO + H2 rich)
           v
    +-------------------------+
    |  Gas Permeation         |---> permeate (CO2 + CO + H2)
    |  Membrane  (mem)        |       * default: H2 in permeate discarded,
    +------------+------------+         CO2+CO recycled to electrolyzer
                 |                    * recycle_permeate_H2=True: permeate
                 v                      feeds a second Polyimide membrane (mem2):
           retentate: CO + H2            mem2 permeate (H2-rich) → credited to
           (CO2 depletion >=              syngas; mem2 retentate (CO2+CO) →
            co2_depletion_target,         recycled to electrolyzer
            default 98 %)

Recycle mole balance (closed loop)
-----------------------------------
    nf_CO2_elec_in = n_CO2_fresh
                     + nf_fl_CO2  [if with_flash]
                     + mem.nf_p_CO2_out
    nf_CO_elec_in  = nf_fl_CO   [if with_flash]
                     + mem.nf_p_CO_out

Design / optimisation variables
---------------------------------
    Electrolyzer : E_appl, v, n (number of cells)
    Compression  : P_inter[k]  (k = 1..n_stages-1),  T_ic_out[k]
    Membrane     : A_mem,  p_p  (permeate pressure)
    System       : n_CO2_fresh  (fresh CO2 feed rate  [mol/s])
    Optional     : purge_fraction  (when float_purge=True)
                   mem2.A_mem, mem2.p_r, comp2_power_kw
                   (when recycle_permeate_H2=True)

CO2 depletion target
--------------------
    nf_r_CO2[N+1] / total_retentate <= 1 - co2_depletion_target/100  (default 98 %)

Objective
---------
    minimise  total_lifecycle_cost = total_capex + plant_life * total_opex_annual

Note on composition for Cp
---------------------------
    MixtureCpModel (inside MultiStageCompressionFlowsheet) uses *fixed*
    nominal mole fractions (z_dict_nominal) for the Shomate Cp integral.
    The actual electrolyzer outlet molar flows are linked through the
    overridden power / duty constraints, so equipment sizing is correct.
    Variations in mixture Cp with composition are typically < 5 % and are
    an accepted engineering approximation for equipment sizing.
"""

import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyomo.environ as pyo

from configuration import Economics, Constants, H2Parameters, PSAParameters
from SubModels.CO2_ElectrolyzerSurrogateModel import CO2Electrolyzer
from SubModels.CompressionFlash.MultiStageCompressionFlowsheet import (
    MultiStageCompressionFlowsheet,
)
from SubModels.CompressionFlash.HeatEXModel import HeatExchangerModel
from SubModels.GasPermeationMembraneModel import GasPermeationMembrane
from SubModels.H2_Electrolyzer_Model import H2Model
from SubModels.solver_utils import run_solver

# ---------------------------------------------------------------------------
# Flash-drum CAPEX constants  (Turton et al. 2012, Table A.1 — vertical vessel)
# Duplicated here to avoid coupling to private FlashModel module variables.
# ---------------------------------------------------------------------------
_FLASH_K1   = 3.4974
_FLASH_K2   = 0.4485
_FLASH_K3   = 0.1074
_FLASH_LN10 = math.log(10)
_FLASH_CEPCI = 797.9 / 397.0          # 2023 vs 2001 basis
_FLASH_FBM   = 2.25 + 1.82 * 1.0 * 2.0  # bare-module factor, 30 bar, CS vessel


class MembraneFlowsheet(pyo.ConcreteModel):
    """
    See module docstring for full description.

    Parameters
    ----------
    n_stages : int
        Number of compression stages (>= 1).
    with_flash : bool
        Include a flash drum between HX2 and the membrane.
    K_dict : dict  {str: float}
        Flash equilibrium constants  {"CO2": ..., "CO": ..., "H2": ...}.
        Required when with_flash=True.
    membrane_material : str
        One of "Polyimide", "PEBAX", "Matrimid", "PRISM".
    N_mem_segments : int
        Number of discretisation segments in the membrane model.
    z_dict_nominal : dict  {str: float}
        Nominal mole fractions used for Shomate Cp calculations in the
        compression flowsheet.  Defaults to a typical electrolyzer outlet
        composition {"CO2": 0.66, "CO": 0.33, "H2": 0.01}.
    feed_flow_nominal : float
        Nominal total molar flow [mol/s] — used for initial guesses.
    Pin_comp : float
        Compression inlet pressure  [bar].
    Pout_comp : float
        Compression outlet pressure [bar].  Also used as membrane p_r.
    T_inter : float
        HX1 outlet / intercooler target temperature [K]  (default 30 °C).
    T_final : float
        Flash drum operating temperature [K].  Used only when with_flash=True;
        an NH3 refrigeration HX2 is created to cool the gas from T_inter to
        T_final before the flash drum.
    gamma : float
        Isentropic exponent for compression.
    eta_comp : float
        Isentropic efficiency of each compressor stage.
    purge_fraction : float
        Fraction of the recycle stream (membrane permeate and, if applicable,
        flash liquid) that is vented to flare rather than recycled [0, 1].
        Default 0.0 (fully closed recycle).  Ignored as a constructor value
        when ``float_purge=True`` — used only as the initial guess for the
        optimisation variable.
    float_purge : bool
        If True, ``purge_fraction`` becomes a Pyomo Var (decision variable)
        instead of a fixed Param, allowing the optimiser to find the optimal
        purge split.  Default False.
    recycle_permeate_H2 : bool
        If True, the membrane permeate (CO2 + CO + H2) is not directly
        recycled.  Instead it is fed — after applying the purge fraction —
        to a second Polyimide membrane (``mem2``) whose H2-rich permeate is
        credited to the syngas product and whose CO2/CO-rich retentate is
        recycled to the electrolyzer.  A dedicated single-stage compressor
        (``comp2``) pressurises the permeate stream from 1 bar to ``mem2.p_r``.
        Default False.
    """

    def __init__(
        self,
        n_stages: int = 2,
        with_flash: bool = False,
        K_dict: dict = None,
        membrane_material: str = "Polyimide",
        N_mem_segments: int = 5,
        co2_depletion_target: float = 98.0,
        z_dict_nominal: dict = None,
        feed_flow_nominal: float = 150.0,
        Pin_comp: float = 1.0,
        Pout_comp: float = 30.0,
        T_inter: float = 303.15,
        T_final: float = 238.15,
        gamma: float = 1.32,
        eta_comp: float = 0.9,
        purge_fraction: float = 0.0,
        float_purge: bool = False,
        recycle_permeate_H2: bool = False,
        dual_membrane: bool = False,
        membrane_material_2: str = "PEBAX",
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if with_flash and K_dict is None:
            raise ValueError("K_dict must be provided when with_flash=True")
        if with_flash and set(K_dict.keys()) != {"CO2", "CO", "H2"}:
            raise ValueError("K_dict must contain exactly the keys 'CO2', 'CO', 'H2'")
        if not 0.0 <= purge_fraction <= 1.0:
            raise ValueError("purge_fraction must be in [0, 1]")
        if dual_membrane and recycle_permeate_H2:
            raise ValueError("dual_membrane and recycle_permeate_H2 are mutually exclusive")

        self.purge_fraction = purge_fraction
        self._float_purge = float_purge
        self._recycle_permeate_H2 = recycle_permeate_H2
        self._dual_membrane = dual_membrane

        if z_dict_nominal is None:
            z_dict_nominal = {"CO2": 0.66, "CO": 0.33, "H2": 0.01}

        _econ      = Economics()
        op_hours   = _econ.operating_hours_per_year
        plant_life = _econ.plant_life_years

        # ==================================================================
        # 1.  CO2 Electrolyzer
        # ==================================================================
        self.elec = CO2Electrolyzer()

        # ==================================================================
        # 2.  Multi-stage compression flowsheet
        #     Built with *nominal* composition for MixtureCpModel.
        #     Flash handled inline in §4 — set with_flash=False here.
        # ==================================================================
        self.comp = MultiStageCompressionFlowsheet(
            z_dict=z_dict_nominal,
            feed_flow=feed_flow_nominal,
            n_stages=n_stages,
            with_flash=False,   # flash built inline below
            Pin=Pin_comp,
            Pout=Pout_comp,
            T_inter=T_inter,
            T_final=T_final,
            gamma=gamma,
            eta=eta_comp,
            float_outlet_pressure=not with_flash,
        )
        # Deactivate standalone objective and lifecycle equations;
        # individual unit CAPEX / OPEX remain active via Guthrie correlations.
        self.comp.obj.deactivate()
        self.comp.eq_lifecycle_cost.deactivate()

        # ==================================================================
        # 3.  Override flow-dependent constraints with actual electrolyzer flow
        #
        #     MolarCompressorModel.molar_flow and HeatExchangerModel.molar_flow
        #     are fixed Params set to feed_flow_nominal at construction time.
        #     We deactivate the corresponding constraints and replace them with
        #     ones that use self.n_total_elec_out (a Pyomo Var linked to the
        #     actual electrolyzer output).
        # ==================================================================
        self.n_total_elec_out = pyo.Var(
            initialize=feed_flow_nominal,
            bounds=(1.0, 1e5),
            doc="Total molar flow leaving electrolyzer cathode [mol/s]",
        )
        self.eq_n_total = pyo.Constraint(
            expr=self.n_total_elec_out
            == self.elec.nf_CO2_elec_out
            + self.elec.nf_CO_elec_out
            + self.elec.nf_H2_elec_out,
        )

        n_comp = len(self.comp._comps)

        # --- Compressor power constraints ---
        for k, comp_k in enumerate(self.comp._comps, start=1):
            cp_mix_k = (
                self.comp.cp_hx.cp_mix
                if k == n_comp
                else self.comp._cp_ic[k - 1].cp_mix
            )
            T_in_k = comp_k.Tin if k == 1 else self.comp._T_ic_out[k - 2]
            getattr(self.comp, f"eq_comp{k}_power").deactivate()
            setattr(
                self,
                f"eq_comp{k}_power_act",
                pyo.Constraint(
                    expr=comp_k.power_kw
                    == self.n_total_elec_out
                    * cp_mix_k
                    * (comp_k.Tout - T_in_k)
                    / 1000.0,
                ),
            )

        # --- Intercooler duty constraints ---
        for k in range(1, n_comp):
            ic_k = self.comp._ics[k - 1]
            cp_k = self.comp._cp_ic[k - 1]
            getattr(self.comp, f"eq_ic{k}_duty").deactivate()
            setattr(
                self,
                f"eq_ic{k}_duty_act",
                pyo.Constraint(
                    expr=ic_k.duty_kw
                    == self.n_total_elec_out * cp_k.delta_h_mix,
                ),
            )

        # --- HX1 duty ---
        self.comp.eq_hx1_duty.deactivate()
        self.eq_hx1_duty_act = pyo.Constraint(
            expr=self.comp.hx1.duty_kw
            == self.n_total_elec_out * self.comp.cp_hx.delta_h_mix,
        )

        # ==================================================================
        # 4.  Flash drum  (inline, variable feed composition)
        #     The existing FlashModel uses fixed Params for z and F_total,
        #     so we build the Rachford-Rice and vessel economics directly here.
        # ==================================================================
        if with_flash:
            # --- Equilibrium K-values (fixed at flash T and P) ---
            self.K_CO2_f = pyo.Param(
                initialize=K_dict["CO2"], doc="Flash K-value CO2 [-]"
            )
            self.K_CO_f = pyo.Param(
                initialize=K_dict["CO"], doc="Flash K-value CO [-]"
            )
            self.K_H2_f = pyo.Param(
                initialize=K_dict["H2"], doc="Flash K-value H2 [-]"
            )

            # --- Feed mole fractions (actual, from electrolyzer outlet) ---
            self.z_f_CO2 = pyo.Var(initialize=0.66, bounds=(1e-8, 1.0))
            self.z_f_CO  = pyo.Var(initialize=0.33, bounds=(1e-8, 1.0))
            self.z_f_H2  = pyo.Var(initialize=0.01, bounds=(1e-8, 1.0))

            self.eq_zf_CO2 = pyo.Constraint(
                expr=self.z_f_CO2 * self.n_total_elec_out
                == self.elec.nf_CO2_elec_out,
            )
            self.eq_zf_CO = pyo.Constraint(
                expr=self.z_f_CO * self.n_total_elec_out
                == self.elec.nf_CO_elec_out,
            )
            self.eq_zf_H2 = pyo.Constraint(
                expr=self.z_f_H2 * self.n_total_elec_out
                == self.elec.nf_H2_elec_out,
            )

            # --- Rachford-Rice: vapour fraction ---
            self.psi_f = pyo.Var(
                bounds=(0.001, 0.999),
                initialize=0.5,
                doc="Flash vapour fraction psi [-]",
            )
            self.eq_rr = pyo.Constraint(
                expr=(
                    self.z_f_CO2
                    * (self.K_CO2_f - 1)
                    / (1 + self.psi_f * (self.K_CO2_f - 1))
                    + self.z_f_CO
                    * (self.K_CO_f - 1)
                    / (1 + self.psi_f * (self.K_CO_f - 1))
                    + self.z_f_H2
                    * (self.K_H2_f - 1)
                    / (1 + self.psi_f * (self.K_H2_f - 1))
                )
                == 0,
            )

            # --- Liquid and vapour mole fractions ---
            self.x_f_CO2 = pyo.Var(initialize=0.90, bounds=(1e-8, 1.0))
            self.x_f_CO  = pyo.Var(initialize=0.09, bounds=(1e-8, 1.0))
            self.x_f_H2  = pyo.Var(initialize=1e-4, bounds=(1e-9, 1.0))
            self.y_f_CO2 = pyo.Var(initialize=0.27, bounds=(1e-8, 1.0))
            self.y_f_CO  = pyo.Var(initialize=0.36, bounds=(1e-8, 1.0))
            self.y_f_H2  = pyo.Var(initialize=0.37, bounds=(1e-8, 1.0))

            self.eq_xCO2 = pyo.Constraint(
                expr=self.x_f_CO2 * (1 + self.psi_f * (self.K_CO2_f - 1))
                == self.z_f_CO2,
            )
            self.eq_xCO = pyo.Constraint(
                expr=self.x_f_CO * (1 + self.psi_f * (self.K_CO_f - 1))
                == self.z_f_CO,
            )
            self.eq_xH2 = pyo.Constraint(
                expr=self.x_f_H2 * (1 + self.psi_f * (self.K_H2_f - 1))
                == self.z_f_H2,
            )
            self.eq_yCO2 = pyo.Constraint(
                expr=self.y_f_CO2 == self.K_CO2_f * self.x_f_CO2
            )
            self.eq_yCO = pyo.Constraint(
                expr=self.y_f_CO == self.K_CO_f * self.x_f_CO
            )
            self.eq_yH2 = pyo.Constraint(
                expr=self.y_f_H2 == self.K_H2_f * self.x_f_H2
            )

            # --- Flash stream molar flows [mol/s] ---
            self.nf_fv_CO2 = pyo.Var(initialize=30.0, bounds=(1e-6, 1e5))
            self.nf_fv_CO  = pyo.Var(initialize=50.0, bounds=(1e-6, 1e5))
            self.nf_fv_H2  = pyo.Var(initialize=1.5,  bounds=(1e-6, 1e5))
            self.nf_fl_CO2 = pyo.Var(initialize=70.0, bounds=(1e-6, 1e5))
            self.nf_fl_CO  = pyo.Var(initialize=0.5,  bounds=(1e-6, 1e5))
            self.nf_fl_H2  = pyo.Var(initialize=1e-3, bounds=(1e-9, 1e5))

            self.eq_nfv_CO2 = pyo.Constraint(
                expr=self.nf_fv_CO2
                == self.n_total_elec_out * self.psi_f * self.y_f_CO2,
            )
            self.eq_nfv_CO = pyo.Constraint(
                expr=self.nf_fv_CO
                == self.n_total_elec_out * self.psi_f * self.y_f_CO,
            )
            self.eq_nfv_H2 = pyo.Constraint(
                expr=self.nf_fv_H2
                == self.n_total_elec_out * self.psi_f * self.y_f_H2,
            )
            self.eq_nfl_CO2 = pyo.Constraint(
                expr=self.nf_fl_CO2
                == self.n_total_elec_out * (1 - self.psi_f) * self.x_f_CO2,
            )
            self.eq_nfl_CO = pyo.Constraint(
                expr=self.nf_fl_CO
                == self.n_total_elec_out * (1 - self.psi_f) * self.x_f_CO,
            )
            self.eq_nfl_H2 = pyo.Constraint(
                expr=self.nf_fl_H2
                == self.n_total_elec_out * (1 - self.psi_f) * self.x_f_H2,
            )

            # --- Flash drum CAPEX  (Turton 2012, vertical vessel, CS, 30 bar) ---
            # Vessel cross-sectional area from Souders-Brown criterion:
            #   A = (F_vap_mol/s × MW_mix  [kg/s]) / (rho_v  [kg/m³] × u_max  [m/s])
            _rho_v  = 57.5    # kg/m³  vapour density at flash conditions
            _u_max  = 0.15    # m/s    Souders-Brown maximum velocity
            _MW_mix = 0.037   # kg/mol approximate mixture molecular weight

            self.flash_area   = pyo.Var(initialize=0.1,   bounds=(1e-4, 200.0))
            self.flash_dia    = pyo.Var(initialize=0.3,   bounds=(0.05, 15.0))
            self.flash_height = pyo.Var(initialize=1.0,   bounds=(0.10, 45.0))
            self.flash_vol    = pyo.Var(initialize=0.1,   bounds=(1e-4, 2000.0))
            self.flash_capex_purch = pyo.Var(initialize=2e3, within=pyo.NonNegativeReals)
            self.flash_CAPEX       = pyo.Var(initialize=3e4, within=pyo.NonNegativeReals)
            self.flash_opex_annual = pyo.Var(initialize=1e3, within=pyo.NonNegativeReals)

            self.eq_flash_area = pyo.Constraint(
                expr=self.flash_area
                == (self.n_total_elec_out * self.psi_f * _MW_mix / _rho_v) / _u_max,
            )
            self.eq_flash_dia = pyo.Constraint(
                expr=self.flash_dia
                == 2.0 * pyo.sqrt(self.flash_area / 3.14159),
            )
            self.eq_flash_height = pyo.Constraint(
                expr=self.flash_height == self.flash_dia * 3.0,
            )
            self.eq_flash_vol = pyo.Constraint(
                expr=self.flash_vol == self.flash_area * self.flash_height,
            )
            self.eq_flash_capex_purch = pyo.Constraint(
                expr=self.flash_capex_purch
                == pyo.exp(
                    (
                        _FLASH_K1
                        + _FLASH_K2 * pyo.log(self.flash_vol) / _FLASH_LN10
                        + _FLASH_K3
                        * (pyo.log(self.flash_vol) / _FLASH_LN10) ** 2
                    )
                    * _FLASH_LN10
                ),
            )
            _f_ind = 1.38
            _eur   = _econ.convert_usd_to_euro
            _maint = _econ.maintenance_factor_static

            self.eq_flash_CAPEX = pyo.Constraint(
                expr=self.flash_CAPEX
                == self.flash_capex_purch
                * _FLASH_CEPCI
                * _FLASH_FBM
                * _f_ind
                * _eur,
            )
            self.eq_flash_opex = pyo.Constraint(
                expr=self.flash_opex_annual == _maint * self.flash_CAPEX,
            )

            # --- HX2 (NH3 refrigeration) + duty with CO2 condensation latent heat ---
            # CO2 latent heat of vaporisation ≈ 11.5 kJ/mol at ~30 bar / 270 K (NIST)
            self.hx2 = HeatExchangerModel(
                Tin=T_inter, Tout=T_final,
                molar_flow=feed_flow_nominal, cp_molar=1.0,
                cooling_medium="NH3", name="HeatExchanger_NH3",
            )
            self.hx2.obj.deactivate()
            # Fix T_cold2 on the shared cp_hx to T_final so delta_h_mix2 = H(T_inter)-H(T_final)
            self.comp.cp_hx.T_cold2.fix(T_final)
            self.delta_H_vap_CO2 = pyo.Param(
                initialize=11.5,
                doc="CO2 condensation latent heat [kJ/mol]  (NIST, ~30 bar/270 K)",
            )
            self.hx2.eq_duty.deactivate()
            self.eq_hx2_duty = pyo.Constraint(
                expr=self.hx2.duty_kw
                == self.n_total_elec_out * self.comp.cp_hx.delta_h_mix2
                + self.nf_fl_CO2 * self.delta_H_vap_CO2,
            )

        # ==================================================================
        # 5.  Gas permeation membrane
        # ==================================================================
        self.mem = GasPermeationMembrane(
            N_segments=N_mem_segments,
            membrane_material=membrane_material,
            # In dual_membrane mode the CO2 depletion target is enforced on stage 2 (mem_s2)
            co2_depletion_target=co2_depletion_target if not dual_membrane else None,
        )
        # Retentate-side pressure = compression outlet pressure
        if with_flash:
            self.mem.p_r.fix(Pout_comp * 1e5)   # Pa — fixed when flash sets the pressure
        else:
            # Outlet pressure floats: link membrane p_r to the optimisation variable
            self.eq_p_r = pyo.Constraint(
                expr=self.mem.p_r == self.comp.Pout_var * 1e5
            )
        self.mem.p_p.fix(1e5)   # Pa — permeate fixed at 1 bar

        # ==================================================================
        # 6.  Connect upstream outlet → membrane feed (node 1)
        # ==================================================================
        if with_flash:
            # Flash vapour feeds the membrane
            self.eq_mem_CO2 = pyo.Constraint(
                expr=self.mem.nf_r_CO2[1] == self.nf_fv_CO2,
            )
            self.eq_mem_CO = pyo.Constraint(
                expr=self.mem.nf_r_CO[1] == self.nf_fv_CO,
            )
            self.eq_mem_H2 = pyo.Constraint(
                expr=self.mem.nf_r_H2[1] == self.nf_fv_H2,
            )
        else:
            # Full electrolyzer outlet feeds the membrane
            self.eq_mem_CO2 = pyo.Constraint(
                expr=self.mem.nf_r_CO2[1] == self.elec.nf_CO2_elec_out,
            )
            self.eq_mem_CO = pyo.Constraint(
                expr=self.mem.nf_r_CO[1] == self.elec.nf_CO_elec_out,
            )
            self.eq_mem_H2 = pyo.Constraint(
                expr=self.mem.nf_r_H2[1] == self.elec.nf_H2_elec_out,
            )

        # ==================================================================
        # 6b.  Second membrane (Polyimide) for permeate H2 recovery
        #
        #      Only active when recycle_permeate_H2=True.
        #
        #      Flow path:
        #        mem1 permeate (CO2+CO+H2 at p_p ≈ 1 bar)
        #          → comp2 (1 bar → mem2.p_r, floating)
        #          → mem2 (Polyimide, H2-selective)
        #            permeate : H2-rich  → credited to syngas H2
        #            retentate: CO2 + CO → recycled back to electrolyzer
        #                       (H2 remaining in retentate is discarded)
        # ==================================================================
        if recycle_permeate_H2:
            _psa = PSAParameters()

            # ---- Second membrane -------------------------------------------
            self.mem2 = GasPermeationMembrane(
                N_segments=N_mem_segments,
                membrane_material="Polyimide",
            )
            self.mem2.p_p.fix(1e5)   # permeate at 1 bar

            _N2 = self.mem2._N

            # Feed connections: post-purge mem1 permeate → mem2 retentate inlet.
            # The purge is applied before compression, so only (1-purge_fraction)
            # of the mem1 permeate enters mem2.
            self.eq_mem2_CO2_feed = pyo.Constraint(
                expr=self.mem2.nf_r_CO2[1] == (1.0 - self.purge_fraction) * self.mem.nf_p_CO2_out
            )
            self.eq_mem2_CO_feed = pyo.Constraint(
                expr=self.mem2.nf_r_CO[1] == (1.0 - self.purge_fraction) * self.mem.nf_p_CO_out
            )
            self.eq_mem2_H2_feed = pyo.Constraint(
                expr=self.mem2.nf_r_H2[1] == (1.0 - self.purge_fraction) * self.mem.nf_p_H2_out
            )

            # ---- Second compressor (1 bar → mem2.p_r) ----------------------
            # Average cp for the permeate mixture (CO2/CO/H2); use same value
            # as the main compressor for consistency.
            _cp2    = 35.0    # J/(mol·K)
            _Tin2   = T_inter  # 303.15 K — assume cooled to intercooler temp

            self.comp2_nf_total = pyo.Var(
                initialize=5.0, bounds=(1e-6, 1e4),
                doc="Total molar flow entering second compressor [mol/s]",
            )
            self.comp2_Tout = pyo.Var(
                initialize=350.0, bounds=(200.0, 800.0),
                doc="Second compressor outlet temperature [K]",
            )
            self.comp2_power_kw = pyo.Var(
                initialize=50.0, bounds=(0.1, 1e5),
                doc="Second compressor shaft power [kW]",
            )
            self.comp2_base_cost = pyo.Var(initialize=2e4, bounds=(0.0, None))
            self.comp2_Fa_cost   = pyo.Var(initialize=3e4, bounds=(0.0, None))
            self.comp2_installed = pyo.Var(initialize=1e5, bounds=(0.0, None))
            self.comp2_capex = pyo.Var(
                initialize=1e5, bounds=(0.0, None),
                doc="Second compressor CAPEX [€]",
            )
            self.comp2_opex = pyo.Var(
                initialize=1e4, bounds=(0.0, None),
                doc="Second compressor annual OPEX [€/yr]",
            )

            # Total feed flow (post-purge): equals sum of mem2 retentate inlet flows,
            # which are already constrained to (1-purge)*mem1_permeate.
            self.eq_comp2_flow = pyo.Constraint(
                expr=self.comp2_nf_total == (
                    self.mem2.nf_r_CO2[1]
                    + self.mem2.nf_r_CO[1]
                    + self.mem2.nf_r_H2[1]
                )
            )

            # Isentropic outlet temperature
            # Tout = Tin + (Tin/eta) * ((Pout/Pin)^((g-1)/g) - 1)
            # Pin = 1e5 Pa (mem.p_p fixed); Pout = mem2.p_r (Pa, floating)
            self.eq_comp2_Tout = pyo.Constraint(
                expr=self.comp2_Tout == _Tin2 + (_Tin2 / eta_comp) * (
                    (self.mem2.p_r / 1e5) ** ((gamma - 1.0) / gamma) - 1.0
                )
            )

            # Shaft power
            self.eq_comp2_power = pyo.Constraint(
                expr=self.comp2_power_kw == (
                    self.comp2_nf_total * _cp2 * (self.comp2_Tout - _Tin2)
                ) / 1000.0
            )

            # Guthrie CAPEX
            _f_inst  = _psa.factor_installation_comp   # 1.565
            _f_ind   = _psa.indirect_cost_factor_comp  # 1.38
            _usd_eur = _econ.convert_usd_to_euro

            self.eq_comp2_base_cost = pyo.Constraint(
                expr=self.comp2_base_cost == 23000.0 * (self.comp2_power_kw / 73.55) ** 0.77
            )
            self.eq_comp2_Fa_cost = pyo.Constraint(
                expr=self.comp2_Fa_cost == 776.5 * (self.comp2_base_cost + 1e-6) ** 0.493
            )
            self.eq_comp2_installed = pyo.Constraint(
                expr=self.comp2_installed == _f_inst * (self.comp2_base_cost + self.comp2_Fa_cost)
            )
            self.eq_comp2_capex = pyo.Constraint(
                expr=self.comp2_capex == _usd_eur * _f_ind * self.comp2_installed
            )
            self.eq_comp2_opex = pyo.Constraint(
                expr=self.comp2_opex == (
                    self.comp2_power_kw * op_hours * _econ.price_power
                    + _econ.maintenance_factor_rotating * self.comp2_capex
                )
            )

        # ==================================================================
        # 6c.  Dual-membrane setup  (only when dual_membrane=True)
        #
        #      Stage 1 (mem):  H2-selective (e.g. Polyimide)
        #        feed     : full electrolyzer outlet  (CO2 + CO + H2 at p_r)
        #        permeate : H2-rich  → credited to syngas H2
        #        retentate: CO2 + CO (H2-depleted, still at p_r) → stage 2 feed
        #
        #      Stage 2 (mem_s2): CO2-selective (e.g. PEBAX)
        #        feed     : stage 1 retentate  (same pressure, no recompressor)
        #        permeate : CO2-rich → recycled to electrolyzer
        #        retentate: CO-rich  → syngas CO product
        #
        #      CO2 depletion target enforced on mem_s2 retentate.
        # ==================================================================
        if dual_membrane:
            self.mem_s2 = GasPermeationMembrane(
                N_segments=N_mem_segments,
                membrane_material=membrane_material_2,
                co2_depletion_target=co2_depletion_target,
            )
            self.mem_s2.p_p.fix(1e5)   # permeate at 1 bar

            # Stage 2 feed pressure: anywhere between 1 bar and mem1 retentate pressure.
            # Pressure let-down via a throttle valve is free (no cost, no energy).
            self.con_p_r_s2_upper = pyo.Constraint(
                expr=self.mem_s2.p_r <= self.mem.p_r
            )
            # Lower bound of 1 bar is already set by GasPermeationMembrane's p_r Var bounds.

            # Connect stage 1 retentate outlet → stage 2 feed
            _N1 = self.mem._N
            self.eq_mem_s2_CO2 = pyo.Constraint(
                expr=self.mem_s2.nf_r_CO2[1] == self.mem.nf_r_CO2[_N1 + 1]
            )
            self.eq_mem_s2_CO = pyo.Constraint(
                expr=self.mem_s2.nf_r_CO[1] == self.mem.nf_r_CO[_N1 + 1]
            )
            self.eq_mem_s2_H2 = pyo.Constraint(
                expr=self.mem_s2.nf_r_H2[1] == self.mem.nf_r_H2[_N1 + 1]
            )

            # H2 purity limit on the recycle stream (stage-2 permeate):
            # at most 1 % H2 by mole so that H2 is captured in stage 1, not lost to recycle.
            # nf_p_H2_out <= 0.01 * (nf_p_CO2_out + nf_p_CO_out + nf_p_H2_out)
            self.con_recycle_H2_limit = pyo.Constraint(
                expr=self.mem_s2.nf_p_H2_out
                <= 0.01 * (
                    self.mem_s2.nf_p_CO2_out
                    + self.mem_s2.nf_p_CO_out
                    + self.mem_s2.nf_p_H2_out
                )
            )

        # ==================================================================
        # 7.  Recycle mole balance  (with optional purge)
        #
        #     A fraction `purge_fraction` of the recycle stream is vented.
        #     purge_fraction=0  → fully closed recycle (original behaviour).
        #     purge_fraction>0  → (1-purge_fraction) of permeate/liquid
        #                         is recycled; the rest is purged to flare.
        #
        #     CO2 recycle : flash liquid CO2  +  membrane permeate CO2
        #     CO  recycle : flash liquid CO   +  membrane permeate CO
        #     H2 in permeate is always discarded (no H2 at electrolyzer inlet).
        # ==================================================================
        if float_purge:
            self.purge_fraction = pyo.Var(
                initialize=purge_fraction,
                bounds=(0.0, 1.0),
                doc="Fraction of recycle stream purged to flare [-] (decision variable)",
            )
        else:
            self.purge_fraction = pyo.Param(
                initialize=purge_fraction,
                doc="Fraction of recycle stream purged to flare [-]",
            )

        self.n_CO2_fresh = pyo.Var(
            initialize=50.0,
            bounds=(0.1, 1e5),
            doc="Fresh CO2 feed entering the system [mol/s]",
        )

        # When purge_fraction is a Var, use a Pyomo expression so the solver
        # sees the symbolic dependency; when it is a Param the expression is
        # identical in form but evaluates to a constant.
        _recycle = 1.0 - self.purge_fraction

        if dual_membrane:
            # Recycle source: stage-2 (CO2-selective) permeate — purely CO2+CO, no H2.
            # Stage-1 permeate H2 goes directly to syngas, never enters the electrolyzer.
            if with_flash:
                _co2_in = self.n_CO2_fresh + _recycle * (self.nf_fl_CO2 + self.mem_s2.nf_p_CO2_out)
                _co_in  = _recycle * (self.nf_fl_CO + self.mem_s2.nf_p_CO_out)
            else:
                _co2_in = self.n_CO2_fresh + _recycle * self.mem_s2.nf_p_CO2_out
                _co_in  = _recycle * self.mem_s2.nf_p_CO_out
        elif recycle_permeate_H2:
            # mem1 permeate no longer goes directly to recycle — it feeds mem2.
            # mem2 retentate (CO2+CO) is the recycle source; H2 in retentate discarded.
            # Flash liquid (if present) is still recycled directly with purge.
            _N2 = self.mem2._N
            if with_flash:
                _co2_in = self.n_CO2_fresh + _recycle * self.nf_fl_CO2 + self.mem2.nf_r_CO2[_N2 + 1]
                _co_in  = _recycle * self.nf_fl_CO + self.mem2.nf_r_CO[_N2 + 1]
            else:
                _co2_in = self.n_CO2_fresh + self.mem2.nf_r_CO2[_N2 + 1]
                _co_in  = self.mem2.nf_r_CO[_N2 + 1]
        else:
            if with_flash:
                _co2_in = self.n_CO2_fresh + _recycle * (self.nf_fl_CO2 + self.mem.nf_p_CO2_out)
                _co_in  = _recycle * (self.nf_fl_CO + self.mem.nf_p_CO_out)
            else:
                _co2_in = self.n_CO2_fresh + _recycle * self.mem.nf_p_CO2_out
                _co_in  = _recycle * self.mem.nf_p_CO_out

        self.eq_recycle_CO2 = pyo.Constraint(
            expr=self.elec.nf_CO2_elec_in == _co2_in,
        )
        self.eq_recycle_CO = pyo.Constraint(
            expr=self.elec.nf_CO_elec_in == _co_in,
        )

        # H2 recycle: permeate H2 (and flash vapour H2 if with_flash) passes through
        # the electrolyzer as an inert diluent and re-enters the compression train.
        if dual_membrane:
            # In dual_membrane mode H2 is extracted at stage 1 — none enters the recycle.
            _h2_in = 0.0
        elif recycle_permeate_H2:
            # mem1 permeate H2 is sent to mem2: its H2-rich permeate is credited
            # to syngas (see §economics), so only the H2 remaining in the mem2
            # retentate recycles to the electrolyzer alongside the CO2+CO.
            # (Without this branch the full mem1 permeate H2 would both feed the
            # cell AND be credited to syngas — double-counting H2.)
            _N2 = self.mem2._N
            _h2_in = self.mem2.nf_r_H2[_N2 + 1]
        elif with_flash:
            _h2_in = _recycle * (self.nf_fl_H2 + self.mem.nf_p_H2_out)
        else:
            _h2_in = _recycle * self.mem.nf_p_H2_out

        self.eq_recycle_H2 = pyo.Constraint(
            expr=self.elec.nf_H2_elec_in == _h2_in,
        )

        # ==================================================================
        # 8.  Economics aggregation  (all OPEX converted to €/yr)
        # ==================================================================
        self.total_capex = pyo.Var(initialize=2e6, bounds=(0.0, None))
        self.total_opex  = pyo.Var(initialize=5e5, bounds=(0.0, None))

        _capex = (
            self.elec.CAPEX
            + self.comp.total_capex   # compressors + intercoolers + HX1 + HX2
            + self.mem.CAPEX
        )
        _opex = (
            self.elec.OPEX * op_hours         # €/h  → €/yr
            + self.comp.total_opex            # already €/yr
            + self.mem.OPEX * op_hours        # €/h  → €/yr
        )
        if with_flash:
            _capex = _capex + self.flash_CAPEX
            _opex  = _opex  + self.flash_opex_annual
        if recycle_permeate_H2:
            _capex = _capex + self.mem2.CAPEX + self.comp2_capex
            _opex  = _opex  + self.mem2.OPEX * op_hours + self.comp2_opex
        if dual_membrane:
            _capex = _capex + self.mem_s2.CAPEX
            _opex  = _opex  + self.mem_s2.OPEX * op_hours

        self.eq_total_capex = pyo.Constraint(expr=self.total_capex == _capex)
        self.eq_total_opex  = pyo.Constraint(expr=self.total_opex  == _opex)

        self.total_lifecycle_cost = pyo.Var(initialize=1e7, bounds=(0.0, None))
        # Discounted lifecycle cost: CAPEX + PVF × OPEX
        # PVF = (1-(1+r)^-n)/r at r=8.5%, n=20 → PVF=9.46  (vs undiscounted 20)
        self.eq_lifecycle_cost = pyo.Constraint(
            expr=self.total_lifecycle_cost
            == self.total_capex + _econ.pvf * self.total_opex,
        )

        # ==================================================================
        # 9.  Objective: minimise total lifecycle cost
        # ==================================================================
        self.obj = pyo.Objective(
            expr=self.total_lifecycle_cost, sense=pyo.minimize,
        )

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def set_syngas_target(
        self,
        syngas_target: float = 40.0,
        syngas_ratio: float = 2.0,
        H2_type: str = "Electrolyzer",
        price_CO2_feed: float = Economics.price_CO2_feed,
        price_H2_buy: float = 0.96 * 3.0 * 2.016 / 1000,
    ):
        """Add syngas production target and H2 sourcing constraints.

        The membrane retentate (CO + H2 + trace CO2) forms the syngas product.
        The H2/CO ratio target is met by one of three sourcing strategies
        controlled by ``H2_type``:

        - ``"Electrolyzer"``   — use only the H2 already present in the
          retentate (produced as a by-product of the CO2 electrolyzer), plus
          the mem2 permeate H2 if ``recycle_permeate_H2=True``. No additional
          H2 equipment or cost is added.
        - ``"Buy"``            — supplement retentate H2 (and mem2 permeate H2
          if applicable) with externally purchased H2. A continuous variable
          ``nf_H2_buy`` [mol/s] is added and its cost appended to
          ``total_opex``.
        - ``"H2Electrolyzer"`` — supplement retentate H2 (and mem2 permeate H2
          if applicable) with H2 produced on-site by a dedicated PEM/alkaline
          water electrolyzer (``H2Model``).  The electrolyzer is sized by the
          optimizer via ``power_hp_mw`` [MW] (lower bound 10 MW, relaxed to
          1 MW here). Its CAPEX is added to ``total_capex`` and its annual
          electricity + maintenance costs are added to ``total_opex``.

        Calling this method also appends feedstock costs (CO2 fresh feed and,
        if applicable, purchased or produced H2) to ``total_opex`` and
        ``total_capex`` so the lifecycle objective already includes them.

        Parameters
        ----------
        syngas_target : float
            Desired syngas mass flow rate [t/day].
        syngas_ratio : float
            Molar H2/CO ratio in the product syngas (e.g. 2 for FT synthesis).
        H2_type : str
            "Electrolyzer"   — use only retentate H2 (no extra equipment).
            "Buy"            — supplement retentate H2 with purchased H2.
            "H2Electrolyzer" — supplement retentate H2 with a dedicated water
                               electrolyzer (H2Model), co-optimized with the
                               rest of the flowsheet.
        price_CO2_feed : float
            Cost of the fresh CO2 feed stream [€/mol].
            Default: 0.05 €/kg × 44.01 g/mol.
        price_H2_buy : float
            Purchase price of H2 [€/mol]  (only used when H2_type="Buy").
            Default: 3 USD/kg × 0.96 USD→EUR × 2.016 g/mol.
        """
        if H2_type not in ("Electrolyzer", "Buy", "H2Electrolyzer"):
            raise ValueError("H2_type must be 'Electrolyzer', 'Buy', or 'H2Electrolyzer'")
        if hasattr(self, "_syngas_target_set"):
            raise RuntimeError("set_syngas_target() has already been called.")

        self._syngas_target_set = True
        self._H2_type = H2_type

        _econ        = Economics()
        op_hours     = _econ.operating_hours_per_year
        seconds_per_year = op_hours * 3600.0   # s/yr
        N_mem        = self.mem._N

        # ------------------------------------------------------------------
        # Back-calculate initial guesses from the syngas target
        # ------------------------------------------------------------------
        _mf_kgs      = syngas_target * 1000.0 / (3600.0 * 24.0)   # kg/s
        _MW_sg       = (syngas_ratio * Constants.MW_H2 + Constants.MW_CO) / (syngas_ratio + 1)
        _nf_CO_ig    = _mf_kgs / _MW_sg / (syngas_ratio + 1)       # mol/s  CO
        _nf_H2_ig    = syngas_ratio * _nf_CO_ig                     # mol/s  H2 total
        # Electrolyzer needs to produce at least nf_CO_ig  of CO;
        # assume 2× CO2 feed to account for recycle and losses
        _nf_total_ig = (_nf_CO_ig + _nf_H2_ig) * 2.0

        # Store so initialize_electrolyzer can auto-detect the correct CO target
        self._nf_CO_init_target = _nf_CO_ig

        # Propagate better guesses into sub-model variables (they are Vars
        # already constructed — set_value does not violate bounds if within range)
        _clip = lambda v, lo, hi: max(lo, min(hi, v))
        self.n_CO2_fresh.set_value(_clip(_nf_CO_ig * 1.5, 0.1, 1e4))
        self.n_total_elec_out.set_value(_clip(_nf_total_ig, 1.0, 1e5))

        # Membrane feed guesses
        self.mem.nf_r_CO2[1].set_value(_clip(_nf_CO_ig * 0.3, 1e-5, 1e5))
        self.mem.nf_r_CO[1].set_value(_clip(_nf_CO_ig * 1.05, 1e-5, 1e5))
        self.mem.nf_r_H2[1].set_value(_clip(_nf_H2_ig * 0.5, 1e-5, 1e5))
        # Retentate outlet guesses
        self.mem.nf_r_CO2[N_mem + 1].set_value(_clip(_nf_CO_ig * 0.005, 1e-5, 1e5))
        self.mem.nf_r_CO[N_mem + 1].set_value(_clip(_nf_CO_ig, 1e-5, 1e5))
        self.mem.nf_r_H2[N_mem + 1].set_value(
            _clip(_nf_H2_ig if H2_type == "Electrolyzer" else _nf_H2_ig * 0.1, 1e-5, 1e5)
        )

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.syngas_ratio_param  = pyo.Param(initialize=syngas_ratio,
                                              doc="H2/CO molar ratio in product syngas [-]")
        # Syngas mass flow target [kg/s]
        _mf_target_kgs = syngas_target * 1000.0 / (3600.0 * 24.0)
        self.mf_syngas_target    = pyo.Param(initialize=_mf_target_kgs, mutable=True,
                                              doc="Syngas mass flow target [kg/s]")

        # ------------------------------------------------------------------
        # Syngas product variables  [mol/s]
        # ------------------------------------------------------------------
        _nf_CO_guess = _nf_CO_ig
        _nf_H2_guess = _nf_H2_ig

        self.nf_CO_syngas  = pyo.Var(initialize=_nf_CO_guess,  bounds=(1e-6, 1e4),
                                      doc="Syngas CO flow [mol/s]")
        self.nf_H2_syngas  = pyo.Var(initialize=_nf_H2_guess,  bounds=(1e-6, 1e4),
                                      doc="Syngas H2 flow [mol/s]")
        self.nf_CO2_syngas = pyo.Var(initialize=1e-3, bounds=(1e-9, 1e4),
                                      doc="Syngas CO2 flow (from retentate) [mol/s]")
        self.nf_syngas     = pyo.Var(initialize=_nf_CO_guess + _nf_H2_guess, bounds=(1e-6, 1e4),
                                      doc="Total syngas molar flow [mol/s]")
        self.mf_syngas     = pyo.Var(initialize=_mf_target_kgs, bounds=(1e-9, 1e3),
                                      doc="Syngas mass flow [kg/s]")

        # ------------------------------------------------------------------
        # Syngas composition constraints
        # ------------------------------------------------------------------
        if self._dual_membrane:
            _N_s2 = self.mem_s2._N
            # CO from stage-2 retentate (dominant) + trace CO that permeated stage 1
            self.eq_CO_syngas = pyo.Constraint(
                expr=self.nf_CO_syngas == self.mem_s2.nf_r_CO[_N_s2 + 1] + self.mem.nf_p_CO_out
            )
            # CO2 impurity = trace CO2 left in stage-2 retentate
            self.eq_CO2_syngas = pyo.Constraint(
                expr=self.nf_CO2_syngas == self.mem_s2.nf_r_CO2[_N_s2 + 1]
            )
            # H2 base: stage-1 permeate H2 (main) + stage-2 retentate H2 (residual)
            _base_H2 = self.mem.nf_p_H2_out + self.mem_s2.nf_r_H2[_N_s2 + 1]
        else:
            # CO comes entirely from stage-1 retentate
            self.eq_CO_syngas = pyo.Constraint(
                expr=self.nf_CO_syngas == self.mem.nf_r_CO[N_mem + 1]
            )
            # CO2 in syngas = trace CO2 left in retentate
            self.eq_CO2_syngas = pyo.Constraint(
                expr=self.nf_CO2_syngas == self.mem.nf_r_CO2[N_mem + 1]
            )
            # H2 base: retentate H2 (+ mem2 permeate H2 if recycle_permeate_H2)
            _base_H2 = (
                self.mem.nf_r_H2[N_mem + 1]
                + (self.mem2.nf_p_H2_out if self._recycle_permeate_H2 else 0.0)
            )

        # H2 sourcing
        if H2_type == "Electrolyzer":
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == _base_H2
            )
        elif H2_type == "Buy":
            self.nf_H2_buy = pyo.Var(
                initialize=max(_nf_H2_guess - 1.0, 0.1),
                bounds=(0.0, 1e4),
                doc="Purchased H2 [mol/s]",
            )
            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas == _base_H2 + self.nf_H2_buy
            )
        else:  # "H2Electrolyzer"
            # Supplemental H2 produced by a dedicated PEM/alkaline water electrolyzer.
            # H2Model sizes itself (via power_hp_mw) to supply exactly the gap between
            # the retentate H2 and the syngas ratio target.
            _nf_H2_supplement = max(_nf_H2_guess - 1.0, 0.1)
            _kwh_per_kg = H2Parameters.actual_kwh_per_kg_h2
            _mw_h2 = Constants.MW_H2
            _power_guess_mw = max(
                _nf_H2_supplement * _mw_h2 * _kwh_per_kg * 3600.0 / 1000.0, 10.0
            )
            self.h2_elec = H2Model()
            # Relax the 10 MW lower bound to 1 MW so the optimizer can freely size
            # the electrolyzer down to near-zero supplement (the 10 MW lb in H2Model
            # is appropriate for a standalone unit but prevents convergence here when
            # the retentate already supplies most of the required H2).
            self.h2_elec.power_hp_mw.setlb(1.0)
            # Override the initial guess for power to match the expected supplement
            self.h2_elec.power_hp_mw.set_value(max(_power_guess_mw, 1.0))

            # Deactivate reporting-only constraints that are not needed in the combined
            # flowsheet and introduce bilinear/division terms that cause IPOPT to fail.
            # Only CAPEX, OPEX, MAINTENANCE, and H2_out_mol_per_sec are actually used
            # by the combined objective and H2 balance constraint.
            for _c in (
                "total_h2_kwh_in_constraint",
                "total_h2_kg_constraint",
                "capex_for_1kg_h2_constraint",
                "annualized_capex_per_kg_h2_constraint",
                "total_annualized_cost_constraint",
                "cost_per_kg_h2_constraint",
            ):
                getattr(self.h2_elec, _c).deactivate()
            # (These six constraints live in H2Model.build_reporting_constraints().)

            self.eq_H2_syngas = pyo.Constraint(
                expr=self.nf_H2_syngas
                == _base_H2 + self.h2_elec.H2_out_mol_per_sec
            )

        # H2/CO ratio
        self.eq_syngas_ratio = pyo.Constraint(
            expr=self.nf_H2_syngas == syngas_ratio * self.nf_CO_syngas
        )

        # Total mole flow
        self.eq_nf_syngas = pyo.Constraint(
            expr=self.nf_syngas
            == self.nf_CO_syngas + self.nf_H2_syngas + self.nf_CO2_syngas
        )

        # Mass flow  [kg/s]:  sum_i  nf_i [mol/s] × MW_i [kg/mol]
        # Mass flow target is CO + H2 only; CO2 is an impurity, not counted
        self.eq_mf_syngas = pyo.Constraint(
            expr=self.mf_syngas
            == (
                self.nf_CO_syngas * Constants.MW_CO
                + self.nf_H2_syngas * Constants.MW_H2
            )
        )

        # Mass-flow target (equality — pin the production rate)
        self.eq_mf_syngas_target = pyo.Constraint(
            expr=self.mf_syngas == self.mf_syngas_target
        )

        # ------------------------------------------------------------------
        # Feedstock costs  [€/yr]
        # ------------------------------------------------------------------
        self.price_CO2_feed_param = pyo.Param(
            initialize=price_CO2_feed,
            doc="CO2 fresh feed price [€/mol]",
        )
        self.cost_CO2_feed = pyo.Var(initialize=1e4, bounds=(0.0, None),
                                      doc="Annual CO2 feed cost [€/yr]")
        self.eq_cost_CO2_feed = pyo.Constraint(
            expr=self.cost_CO2_feed
            == self.price_CO2_feed_param * self.n_CO2_fresh * seconds_per_year
        )

        _feedstock_opex = self.cost_CO2_feed

        if H2_type == "Buy":
            self.price_H2_buy_param = pyo.Param(
                initialize=price_H2_buy,
                doc="Purchased H2 price [€/mol]",
            )
            self.cost_H2_buy = pyo.Var(initialize=1e4, bounds=(0.0, None),
                                        doc="Annual purchased H2 cost [€/yr]")
            self.eq_cost_H2_buy = pyo.Constraint(
                expr=self.cost_H2_buy
                == self.price_H2_buy_param * self.nf_H2_buy * seconds_per_year
            )
            _feedstock_opex = _feedstock_opex + self.cost_H2_buy

        # ------------------------------------------------------------------
        # Update total_opex to include feedstock costs (and H2 electrolyzer if used)
        # ------------------------------------------------------------------
        with_flash = hasattr(self, "psi_f")
        _base_opex = (
            self.elec.OPEX * op_hours
            + self.comp.total_opex
            + self.mem.OPEX * op_hours
        )
        if with_flash:
            _base_opex = _base_opex + self.flash_opex_annual
        if self._recycle_permeate_H2:
            _base_opex = _base_opex + self.mem2.OPEX * op_hours + self.comp2_opex
        if self._dual_membrane:
            _base_opex = _base_opex + self.mem_s2.OPEX * op_hours
        if H2_type == "H2Electrolyzer":
            # h2_elec.OPEX and h2_elec.MAINTENANCE are already in €/year
            _base_opex = _base_opex + self.h2_elec.OPEX + self.h2_elec.MAINTENANCE

        self.eq_total_opex.deactivate()
        self.eq_total_opex_syngas = pyo.Constraint(
            expr=self.total_opex == _base_opex + _feedstock_opex
        )

        # ------------------------------------------------------------------
        # Update total_capex to include H2 electrolyzer CAPEX if used
        # ------------------------------------------------------------------
        if H2_type == "H2Electrolyzer":
            self.eq_total_capex.deactivate()
            _capex_with_h2elec = (
                self.elec.CAPEX
                + self.comp.total_capex
                + self.mem.CAPEX
                + self.h2_elec.CAPEX
            )
            if with_flash:
                _capex_with_h2elec = _capex_with_h2elec + self.flash_CAPEX
            if self._recycle_permeate_H2:
                _capex_with_h2elec = _capex_with_h2elec + self.mem2.CAPEX + self.comp2_capex
            if self._dual_membrane:
                _capex_with_h2elec = _capex_with_h2elec + self.mem_s2.CAPEX
            self.eq_total_capex_syngas = pyo.Constraint(
                expr=self.total_capex == _capex_with_h2elec
            )
        # total_lifecycle_cost constraint uses total_opex/total_capex unchanged — no edit needed.

    def initialize_electrolyzer(
        self,
        E_appl: float = -1.31,
        v: float = 0.08,
        CO2share: float = 0.80,
        nf_CO_out_target: float = None,
    ):
        """Set all model variables to a self-consistent starting point.

        Evaluates the surrogate equations in pure Python (no solver required)
        to get concentrations consistent with (E_appl, v, CO2share), then
        seeds every variable in the model — including all intermediate membrane
        segment nodes — so IPOPT's first Jacobian evaluation is numerically
        well-behaved.

        **Must be called before solve().**

        Parameters
        ----------
        E_appl : float
            Applied voltage [V], typically -1.2 to -1.5 (negative).
        v : float
            Gas flow velocity in the electrolyzer channel [m/s].
        CO2share : float
            Inlet CO2 mole fraction [-], i.e. CO2/(CO2+CO) at cathode inlet.
        nf_CO_out_target : float or None
            Desired CO outlet flow [mol/s] used to scale all molar flows.
            If None (default), auto-detected from set_syngas_target() when
            previously called, otherwise falls back to 15.0 mol/s.
        """
        # Auto-detect from set_syngas_target() if not explicitly provided
        if nf_CO_out_target is None:
            nf_CO_out_target = getattr(self, "_nf_CO_init_target", 15.0)

        # Scale membrane area proportional to target CO flow.
        # Reference: A_mem default of 5000 m² is appropriate for ~14.4 mol/s CO (40 t/day).
        # Without this, IPOPT starts from the same 5000 m² regardless of target and can
        # drift into a degenerate stage_cut=0.99 corner for larger targets.
        _A_mem_scaled = max(min(5000.0 * nf_CO_out_target / 14.4, 1e6), 100.0)
        self.mem.A_mem.set_value(_A_mem_scaled)

        _P     = 1e5    # Pa
        _R     = 8.314  # J/(mol·K)
        _T     = 298.0  # K
        _henry = 0.85

        c_total  = _P / (_R * _T)        # ≈ 40.9 mol/m³

        # --- Ensure CO2share is large enough for the CO_ineq constraint (c_out_CO >= c_in_CO).
        # The CO surrogate gives c_out_CO = CO2share * inner^10.43.
        # For c_out_CO > c_in_CO = c_total*(1-CO2share) we need:
        #   CO2share > c_total / (inner^10.43 + c_total)
        # Compute the CO inner expression at the given (E_appl, v):
        try:
            _denom_co   = E_appl - v ** (-0.46066585)
            _base2_co   = E_appl * -0.6278232
            _term2_co   = _base2_co ** 21.189863 if _base2_co >= 0 else -((-_base2_co) ** 21.189863)
            _inner_co   = 1.4358697 / _denom_co - _term2_co + E_appl * -1.1215197
            _c_co_unit  = max(_inner_co, 1e-4) ** 10.42661   # c_out_CO per unit of CO2share
        except Exception:
            _c_co_unit  = 1.5 * c_total   # fallback
        _co2share_min = c_total / max(_c_co_unit + c_total, 1e-9)
        if CO2share < _co2share_min + 0.02:
            CO2share = min(_co2share_min + 0.03, 0.99)

        c_in_CO2 = c_total * CO2share
        c_in_CO  = c_total * (1.0 - CO2share)

        # --- Surrogate equations ---
        c_out_CO2 = (
            (E_appl * (((v ** 0.25 + E_appl) * -3.7203176) - 1.3941393)) ** 3
            + 40.244705
        ) * CO2share
        c_out_CO2 = max(min(c_out_CO2, c_in_CO2 * 0.999), 1e-3)

        try:
            _denom   = E_appl - v ** (-0.46066585)
            _base2   = E_appl * -0.6278232
            _term2   = _base2 ** 21.189863 if _base2 >= 0 else -((-_base2) ** 21.189863)
            _inner   = 1.4358697 / _denom - _term2 + E_appl * -1.1215197
            c_out_CO = CO2share * (max(_inner, 1e-4) ** 10.42661)
        except Exception:
            c_out_CO = c_in_CO * 1.5
        c_out_CO = max(c_out_CO, c_in_CO + 1e-3)

        _base_h2 = (E_appl ** 2 * 0.26091927) + 0.35308436
        c_out_H2 = (max(_base_h2, 1e-6) ** 14.994374) / v
        c_out_H2 = max(c_out_H2, 1e-3)

        c_mean_CO2 = (c_in_CO2 + c_out_CO2) / 2.0 * _henry

        vol_flow   = nf_CO_out_target / max(c_out_CO, 1e-6)
        nf_CO2_out = c_out_CO2 * vol_flow
        nf_CO_out  = c_out_CO  * vol_flow
        nf_H2_out  = c_out_H2  * vol_flow
        nf_total   = nf_CO2_out + nf_CO_out + nf_H2_out
        nf_CO2_in  = c_in_CO2 * vol_flow
        nf_CO_in   = c_in_CO  * vol_flow

        # --- Electrolyzer Vars ---
        self.elec.E_appl.set_value(E_appl)
        self.elec.v.set_value(v)
        self.elec.CO2share.set_value(CO2share)
        self.elec.c_elec_in_CO2.set_value(c_in_CO2)
        self.elec.c_elec_in_CO.set_value(c_in_CO)
        self.elec.c_elec_out_CO2.set_value(c_out_CO2)
        self.elec.c_elec_out_CO.set_value(c_out_CO)
        self.elec.c_elec_out_H2.set_value(c_out_H2)
        self.elec.c_elec_mean_CO2.set_value(c_mean_CO2)
        self.elec.vol_gas_flow.set_value(vol_flow)
        self.elec.nf_CO2_elec_in.set_value(nf_CO2_in)
        self.elec.nf_CO_elec_in.set_value(nf_CO_in)
        self.elec.nf_CO2_elec_out.set_value(nf_CO2_out)
        self.elec.nf_CO_elec_out.set_value(nf_CO_out)
        self.elec.nf_H2_elec_out.set_value(nf_H2_out)

        # --- Electrolyzer electrical variables ---
        # Compute Butler-Volmer quantities consistent with (E_appl, v, c_mean_CO2)
        from math import asinh as _asinh, exp as _exp
        _BV_iCO   = pyo.value(self.elec.BV_iCO)
        _BV_iH2   = pyo.value(self.elec.BV_iH2)
        _BV_CO    = pyo.value(self.elec.BV_CO)
        _BV_H2    = pyo.value(self.elec.BV_H2)
        _BV_ECO   = pyo.value(self.elec.BV_ECO)
        _BV_CO2ref = pyo.value(self.elec.BV_CO2ref)
        _E_anode  = pyo.value(self.elec.E_anode)
        _Ch_H     = pyo.value(self.elec.Channel_H)
        _Ch_W     = pyo.value(self.elec.Channel_W)
        _sigma_el = pyo.value(self.elec.sigma_el)
        _Lm       = pyo.value(self.elec.Lm)
        _sigma_m  = pyo.value(self.elec.sigma_m)
        _r_area   = pyo.value(self.elec.reacting_area)

        _F_const = 96485.0
        _CD = max(
            _BV_iCO * c_mean_CO2 / _BV_CO2ref
            * _exp(-(_BV_CO * _F_const / (_R * _T)) * (E_appl + _BV_ECO))
            + _BV_iH2 * _exp(-(_BV_H2 * _F_const / (_R * _T)) * E_appl),
            1.0
        )
        _eta_actA = (_R * _T) / (0.5 * 96485.0) * _asinh(_CD / (2.0 * 1e-7))
        _eta_ohm  = _CD * (_Ch_H / _sigma_el + _Lm / _sigma_m)
        _Vcell    = _E_anode + _eta_actA + _BV_ECO - E_appl + _eta_ohm
        _n_cells  = vol_flow / max(v * _Ch_H * _Ch_W, 1e-15)
        _tot_area = max(_n_cells * _r_area, 3000.0)   # respect lower bound
        _I        = _CD * _tot_area
        _Power    = _I * _Vcell                        # W

        from configuration import Economics as _Econ
        _price_pw = _Econ.price_power
        _maint_f   = pyo.value(self.elec.maint_factor)
        _b_cost    = pyo.value(self.elec.base_cost_elec)         # €/m²
        _f_inst    = pyo.value(self.elec.f_installation_elec)   # 5.04
        _repl_f_co2 = 1.0 + (1.0 + _Econ.interest_rate) ** (-10)
        _CAPEX_elec = _f_inst * _b_cost * _tot_area * _repl_f_co2  # installed € incl. replacement
        _OPEX_elec = (_Power * _price_pw / 1000.0          # electricity €/h
                      + _maint_f * _CAPEX_elec / _Econ.operating_hours_per_year)  # maintenance €/h

        self.elec.CD.set_value(max(_CD, 1.0))
        self.elec.eta_actA.set_value(_eta_actA)
        self.elec.eta_ohm.set_value(_eta_ohm)
        self.elec.Vcell.set_value(max(_Vcell, 2.5))
        self.elec.n.set_value(max(_n_cells / 1e6, 3.0))
        self.elec.total_area.set_value(max(_tot_area, 3000.0))
        self.elec.I.set_value(max(_I, 1.0))
        self.elec.Power.set_value(max(_Power, 1.0))
        self.elec.OPEX.set_value(max(_OPEX_elec, 1.0))
        self.elec.CAPEX.set_value(max(_CAPEX_elec, 1.0))

        # FE, n_CO2_theoretical, lamda_CO2
        _FE_val = (
            _BV_iCO * c_out_CO2 * _henry / _BV_CO2ref
            * _exp(-(_BV_CO * 96485.0 / (_R * _T)) * (E_appl + _BV_ECO))
            / max(_CD, 1e-9) * 100.0
        )
        self.elec.FE.set_value(max(min(_FE_val, 100.0), 1e-6))
        _n_CO2_th = max(_I / (2.0 * 96485.0), 1e-6)
        self.elec.n_CO2_theoretical.set_value(_n_CO2_th)
        self.elec.lamda_CO2.set_value(
            max(min(nf_CO2_in / max(_n_CO2_th, 1e-9), 6.0), 1.0)
        )

        # --- Bridge and recycle ---
        self.n_total_elec_out.set_value(nf_total)
        nf_CO2_consumed = max(nf_CO2_in - nf_CO2_out, 0.1)
        # Initial guess for fresh CO2: consumed CO2 minus what comes back via recycle.
        # Using nf_CO2_consumed as a proxy (a fraction permeates and returns).
        self.n_CO2_fresh.set_value(nf_CO2_consumed)

        # --- Membrane feed flows ---
        with_flash = hasattr(self, "psi_f")
        # Set mem.p_r early so membrane flux constraints use the correct pressure
        # (must come before the segment-by-segment flux seeding below)
        if not with_flash and hasattr(self, "comp") and hasattr(self.comp, "Pout_var"):
            self.mem.p_r.set_value(pyo.value(self.comp.Pout_var) * 1e5)
        if not with_flash:
            _mem_CO2_feed = nf_CO2_out
            _mem_CO_feed  = nf_CO_out
            _mem_H2_feed  = nf_H2_out
        else:
            # Flash: CO2 mostly condenses → little CO2 in vapour to membrane
            self.z_f_CO2.set_value(nf_CO2_out / nf_total)
            self.z_f_CO.set_value(nf_CO_out / nf_total)
            self.z_f_H2.set_value(nf_H2_out / nf_total)
            self.psi_f.set_value(0.5)
            _mem_CO2_feed = nf_CO2_out * 0.30
            _mem_CO_feed  = nf_CO_out  * 0.95
            _mem_H2_feed  = nf_H2_out  * 0.90
            # Flash stream vars
            self.nf_fv_CO2.set_value(_mem_CO2_feed)
            self.nf_fv_CO.set_value(_mem_CO_feed)
            self.nf_fv_H2.set_value(_mem_H2_feed)
            self.nf_fl_CO2.set_value(nf_CO2_out * 0.70)
            self.nf_fl_CO.set_value(nf_CO_out   * 0.05)
            self.nf_fl_H2.set_value(nf_H2_out   * 0.10)
            self.x_f_CO2.set_value(0.90)
            self.x_f_CO.set_value(0.09)
            self.x_f_H2.set_value(0.01)
            self.y_f_CO2.set_value(0.27)
            self.y_f_CO.set_value(0.36)
            self.y_f_H2.set_value(0.37)

        # --- Membrane: all nodes and segment variables ---
        # Retentate CO2 drops from feed to ~0.4% of feed (purity target);
        # CO and H2 stay ≈ constant (small permeation).
        N_mem = self.mem._N

        _ret_CO2_out = _mem_CO2_feed * 0.004
        _ret_CO_out  = _mem_CO_feed  * 0.95
        _ret_H2_out  = _mem_H2_feed  * 0.95

        # Per-segment flux: total removal divided equally across N segments
        _J_CO2_seg = (_mem_CO2_feed - _ret_CO2_out) / N_mem
        _J_CO_seg  = (_mem_CO_feed  - _ret_CO_out)  / N_mem
        _J_H2_seg  = (_mem_H2_feed  - _ret_H2_out)  / N_mem

        # Retentate flow nodes: linear interpolation feed → retentate outlet
        for k in range(1, N_mem + 2):
            frac = (k - 1) / N_mem          # 0 at k=1, 1 at k=N+1
            self.mem.nf_r_CO2[k].set_value(
                _mem_CO2_feed + frac * (_ret_CO2_out - _mem_CO2_feed)
            )
            self.mem.nf_r_CO[k].set_value(
                _mem_CO_feed  + frac * (_ret_CO_out  - _mem_CO_feed)
            )
            self.mem.nf_r_H2[k].set_value(
                _mem_H2_feed  + frac * (_ret_H2_out  - _mem_H2_feed)
            )

        # Segment mole fractions and fluxes
        for k in range(1, N_mem + 1):
            # Retentate mole fractions at segment k: use upstream node k
            _nf_tot_k = (
                self.mem.nf_r_CO2[k].value
                + self.mem.nf_r_CO[k].value
                + self.mem.nf_r_H2[k].value
            )
            _y_r_CO2 = self.mem.nf_r_CO2[k].value / max(_nf_tot_k, 1e-9)
            _y_r_CO  = self.mem.nf_r_CO[k].value  / max(_nf_tot_k, 1e-9)
            _y_r_H2  = self.mem.nf_r_H2[k].value  / max(_nf_tot_k, 1e-9)
            self.mem.y_r_CO2[k].set_value(max(_y_r_CO2, 1e-8))
            self.mem.y_r_CO[k].set_value(max(_y_r_CO,  1e-8))
            self.mem.y_r_H2[k].set_value(max(_y_r_H2,  1e-8))

            # Fluxes
            _j_co2 = max(_J_CO2_seg, 1e-9)
            _j_co  = max(_J_CO_seg,  1e-9)
            _j_h2  = max(_J_H2_seg,  1e-9)
            self.mem.J_CO2[k].set_value(_j_co2)
            self.mem.J_CO[k].set_value(_j_co)
            self.mem.J_H2[k].set_value(_j_h2)

            # Permeate mole fractions at segment k
            _j_tot = _j_co2 + _j_co + _j_h2
            self.mem.y_p_CO2[k].set_value(max(_j_co2 / _j_tot, 1e-8))
            self.mem.y_p_CO[k].set_value(max(_j_co  / _j_tot, 1e-8))
            self.mem.y_p_H2[k].set_value(max(_j_h2  / _j_tot, 1e-8))

        # Permeate outlet totals
        _perm_CO2_init = max(_J_CO2_seg * N_mem, 1e-6)
        _perm_CO_init  = max(_J_CO_seg  * N_mem, 1e-6)
        _perm_H2_init  = max(_J_H2_seg  * N_mem, 1e-6)
        self.mem.nf_p_CO2_out.set_value(_perm_CO2_init)
        self.mem.nf_p_CO_out.set_value(_perm_CO_init)
        self.mem.nf_p_H2_out.set_value(_perm_H2_init)

        # --- Close recycle constraints ---
        _recycle_frac = 1.0 - pyo.value(self.purge_fraction)
        _nf_CO2_recycle = _recycle_frac * _perm_CO2_init
        _nf_CO_recycle  = _recycle_frac * _perm_CO_init
        _nf_H2_recycle  = _recycle_frac * _perm_H2_init if not self._dual_membrane else 0.0
        self.elec.nf_H2_elec_in.set_value(max(_nf_H2_recycle, 0.0))
        self.n_total_elec_out.set_value(nf_total + _nf_H2_recycle)
        # Fresh CO2 = total CO2 entering electrolyzer minus what comes back via recycle
        _nf_fresh_CO2 = max(pyo.value(self.elec.nf_CO2_elec_in) - _nf_CO2_recycle, 0.1)
        self.n_CO2_fresh.set_value(_nf_fresh_CO2)
        # CO entering electrolyzer must equal recycle CO
        self.elec.nf_CO_elec_in.set_value(max(_nf_CO_recycle, 1e-6))
        self.elec.c_elec_in_CO.set_value(
            max(_nf_CO_recycle / max(vol_flow, 1e-9), 1e-9)
        )
        # Update CO2share at inlet to reflect new recycle composition
        _nf_CO2_in_val = pyo.value(self.elec.nf_CO2_elec_in)
        _co2share_in = _nf_CO2_in_val / max(_nf_CO2_in_val + max(_nf_CO_recycle, 1e-9), 1e-9)
        self.elec.CO2share.set_value(max(min(_co2share_in, 0.9999), 0.01))

        # Seed A_mem to satisfy J_CO2_def in an average sense.
        # Uniform J_CO2[k] = _J_CO2_seg (from linear retentate interpolation) exactly
        # satisfies balance_CO2[k]; A_mem is chosen so the average J_CO2_def holds.
        _permeance_CO2 = pyo.value(self.mem.Q_CO2) / pyo.value(self.mem.delta)
        _p_r_val = pyo.value(self.mem.p_r)
        _p_p_val = pyo.value(self.mem.p_p)
        _df_CO2_list = []
        for _k in range(1, N_mem + 1):
            _df = (pyo.value(self.mem.y_r_CO2[_k]) * _p_r_val
                   - pyo.value(self.mem.y_p_CO2[_k]) * _p_p_val)
            _df_CO2_list.append(max(_df, 1.0))
        _sum_df_CO2 = sum(_df_CO2_list)
        # A_mem from total CO2 removal: sum_k(J_CO2[k]) = permeance * A_mem / N * sum(df_k)
        _CO2_removal = max(_mem_CO2_feed - _ret_CO2_out, 1e-6)
        _A_mem_init = _CO2_removal * N_mem / max(_permeance_CO2 * _sum_df_CO2, 1e-12)
        self.mem.A_mem.set_value(max(min(_A_mem_init, 1e7), 1.0))

        # Stage cut
        _perm_total = (
            self.mem.nf_p_CO2_out.value
            + self.mem.nf_p_CO_out.value
            + self.mem.nf_p_H2_out.value
        )
        _feed_total = _mem_CO2_feed + _mem_CO_feed + _mem_H2_feed
        self.mem.stage_cut.set_value(
            max(min(_perm_total / max(_feed_total, 1e-9), 0.99), 0.01)
        )

        # --- Syngas product variables ---
        if hasattr(self, "_syngas_target_set"):
            N_m = self.mem._N
            self.nf_CO_syngas.set_value(self.mem.nf_r_CO[N_m + 1].value)
            self.nf_CO2_syngas.set_value(self.mem.nf_r_CO2[N_m + 1].value)
            _h2_ret    = self.mem.nf_r_H2[N_m + 1].value
            _h2_needed = pyo.value(self.syngas_ratio_param) * self.mem.nf_r_CO[N_m + 1].value
            self.nf_H2_syngas.set_value(_h2_needed)
            self.nf_syngas.set_value(
                self.mem.nf_r_CO[N_m + 1].value
                + _h2_needed
                + self.mem.nf_r_CO2[N_m + 1].value
            )
            if self._H2_type == "Buy":
                self.nf_H2_buy.set_value(max(_h2_needed - _h2_ret, 0.1))
            elif self._H2_type == "H2Electrolyzer":
                # Back-calculate required power from the supplemental H2 needed
                _h2_supplement = max(_h2_needed - _h2_ret, 0.0)
                _kwh_per_kg = pyo.value(self.h2_elec.actual_kwh_per_kg_h2)
                _mw_h2 = pyo.value(self.h2_elec.MW_H2)
                _power_mw = max(
                    _h2_supplement * _mw_h2 * _kwh_per_kg * 3600.0 / 1000.0, 1.0
                )
                self.h2_elec.power_hp_mw.set_value(_power_mw)
                import math as _math
                self.h2_elec.x_hp.set_value(_math.log(_power_mw))
                _kg_per_s = _h2_supplement * _mw_h2
                self.h2_elec.H2_out_kg_per_sec.set_value(max(_kg_per_s, 1e-6))
                self.h2_elec.H2_out_mol_per_sec.set_value(max(_h2_supplement, 1e-6))
                _total_kwh_in = _power_mw * 1000.0 * pyo.value(self.h2_elec.operating_hours_per_year) * pyo.value(self.h2_elec.plant_life_years)
                self.h2_elec.total_h2_kwh_in.set_value(_total_kwh_in)
                _total_kg = _total_kwh_in / _kwh_per_kg
                self.h2_elec.total_h2_kg.set_value(max(_total_kg, 1.0))
                _CE_PCI = H2Parameters.CE_PCI
                _C_F_el = H2Parameters.C_F_el
                _capex_per_kw = (-38.131 * _math.log(_power_mw)**3
                                 + 433.69 * _math.log(_power_mw)**2
                                 - 1690.0 * _math.log(_power_mw) + 3634.6) * (
                    _CE_PCI / 708.8 * _C_F_el
                )
                self.h2_elec.capex_electrolyzer_per_kw.set_value(_capex_per_kw)
                _repl_f = 1.0 + (1.0 + Economics().interest_rate) ** (-10)
                _capex = _capex_per_kw * _power_mw * 1000.0 * _repl_f
                self.h2_elec.CAPEX.set_value(max(_capex, 1.0))
                _econ_h2 = Economics()
                _opex = _econ_h2.operating_hours_per_year * _power_mw * _econ_h2.price_power * 1000.0
                self.h2_elec.OPEX.set_value(max(_opex, 1.0))
                _maint_f = _econ_h2.maintenance_factor_membrane
                _maintenance = _maint_f * _capex
                self.h2_elec.MAINTENANCE.set_value(max(_maintenance, 0.0))
                _capex_per_kg = _capex / max(_total_kg, 1.0)
                self.h2_elec.capex_for_1kg_h2.set_value(_capex_per_kg)
                _ann_factor = _econ_h2.annuity_factor_plant_life
                _ann_capex = _capex_per_kg / _ann_factor
                self.h2_elec.annualized_capex_per_kg_h2.set_value(_ann_capex)
                _op_hours = _econ_h2.operating_hours_per_year
                _total_cost = _ann_capex * max(_kg_per_s, 1e-9) * 3600.0 * _op_hours + _opex + _maintenance
                self.h2_elec.total_annualized_cost.set_value(max(_total_cost, 0.0))
                _cost_per_kg = _total_cost / max(_kg_per_s * 3600.0 * _op_hours, 1e-9)
                self.h2_elec.cost_per_kg_h2.set_value(_cost_per_kg)

        # --- Cost variable initialisation ---
        from configuration import Constants as _Const
        _Econ_inst  = _Econ()
        _op_hours   = _Econ_inst.operating_hours_per_year
        _plant_life = _Econ_inst.plant_life_years

        # Membrane cost (from default A_mem and BC_membrane)
        _A_mem_val  = pyo.value(self.mem.A_mem)
        _BC_mem     = pyo.value(self.mem.BC_membrane)
        _t_life_mem = pyo.value(self.mem.t_lifetime_membrane)
        _maint_mem  = pyo.value(self.mem.maint_factor)
        _mem_CAPEX  = _BC_mem * _A_mem_val
        _mem_OPEX   = _mem_CAPEX / (_t_life_mem * _op_hours) + _maint_mem * _mem_CAPEX / _op_hours
        self.mem.CAPEX.set_value(max(_mem_CAPEX, 1.0))
        self.mem.OPEX.set_value(max(_mem_OPEX, 1e-3))

        # --- Seed second stage membrane (dual_membrane mode) ---
        if self._dual_membrane:
            _N1 = self.mem._N
            # Stage 2 feed = stage 1 retentate outlet
            _s2_CO2 = max(self.mem.nf_r_CO2[_N1 + 1].value, 1e-6)
            _s2_CO  = max(self.mem.nf_r_CO[_N1 + 1].value,  1e-6)
            _s2_H2  = max(self.mem.nf_r_H2[_N1 + 1].value,  1e-6)
            _s2_total = _s2_CO2 + _s2_CO + _s2_H2
            _N2 = self.mem_s2._N

            # PEBAX is CO2-selective: most CO2 permeates, CO stays in retentate
            _ret_s2_CO2 = max(_s2_CO2 * 0.02, 1e-6)
            _ret_s2_CO  = max(_s2_CO  * 0.97, 1e-6)
            _ret_s2_H2  = max(_s2_H2  * 0.90, 1e-6)

            for _k in range(1, _N2 + 2):
                _frac = (_k - 1) / _N2
                self.mem_s2.nf_r_CO2[_k].set_value(_s2_CO2 + _frac * (_ret_s2_CO2 - _s2_CO2))
                self.mem_s2.nf_r_CO[_k].set_value(_s2_CO  + _frac * (_ret_s2_CO  - _s2_CO))
                self.mem_s2.nf_r_H2[_k].set_value(_s2_H2  + _frac * (_ret_s2_H2  - _s2_H2))

            _y0_CO2 = max(_s2_CO2 / _s2_total, 1e-8)
            _y0_CO  = max(_s2_CO  / _s2_total, 1e-8)
            _y0_H2  = max(_s2_H2  / _s2_total, 1e-8)
            for _k in range(1, _N2 + 1):
                self.mem_s2.y_r_CO2[_k].set_value(_y0_CO2)
                self.mem_s2.y_r_CO[_k].set_value(_y0_CO)
                self.mem_s2.y_r_H2[_k].set_value(_y0_H2)
                _yp_CO2 = min(max(_y0_CO2 * 3.0, 1e-8), 0.98)
                self.mem_s2.y_p_CO2[_k].set_value(_yp_CO2)
                self.mem_s2.y_p_CO[_k].set_value(max((1.0 - _yp_CO2) * 0.8, 1e-8))
                self.mem_s2.y_p_H2[_k].set_value(max((1.0 - _yp_CO2) * 0.2, 1e-8))
                self.mem_s2.J_CO2[_k].set_value(max((_s2_CO2 - _ret_s2_CO2) / _N2, 1e-9))
                self.mem_s2.J_CO[_k].set_value(max((_s2_CO  - _ret_s2_CO)  / _N2, 1e-9))
                self.mem_s2.J_H2[_k].set_value(max((_s2_H2  - _ret_s2_H2)  / _N2, 1e-9))

            self.mem_s2.nf_p_CO2_out.set_value(max(_s2_CO2 - _ret_s2_CO2, 1e-6))
            self.mem_s2.nf_p_CO_out.set_value(max(_s2_CO  - _ret_s2_CO,  1e-6))
            self.mem_s2.nf_p_H2_out.set_value(max(_s2_H2  - _ret_s2_H2,  1e-6))
            _perm_s2_total = (_s2_CO2 - _ret_s2_CO2) + (_s2_CO - _ret_s2_CO) + (_s2_H2 - _ret_s2_H2)
            self.mem_s2.stage_cut.set_value(max(min(_perm_s2_total / max(_s2_total, 1e-9), 0.99), 0.01))
            # Start at half of mem1 retentate pressure as a neutral initial guess
            self.mem_s2.p_r.set_value(max(pyo.value(self.mem.p_r) * 0.5, 1e5))

            _A_s2 = max(_A_mem_val * 0.5, 100.0)
            self.mem_s2.A_mem.set_value(_A_s2)
            _s2_CAPEX = _BC_mem * _A_s2
            _s2_OPEX  = _s2_CAPEX / (_t_life_mem * _op_hours) + _maint_mem * _s2_CAPEX / _op_hours
            self.mem_s2.CAPEX.set_value(max(_s2_CAPEX, 1.0))
            self.mem_s2.OPEX.set_value(max(_s2_OPEX, 1e-3))

        # --- Seed second membrane + compressor (if active) ---
        if self._recycle_permeate_H2:
            _N2 = self.mem2._N
            _purge_val   = pyo.value(self.purge_fraction)
            _p2_feed_CO2 = max((1.0 - _purge_val) * pyo.value(self.mem.nf_p_CO2_out), 1e-6)
            _p2_feed_CO  = max((1.0 - _purge_val) * pyo.value(self.mem.nf_p_CO_out),  1e-6)
            _p2_feed_H2  = max((1.0 - _purge_val) * pyo.value(self.mem.nf_p_H2_out),  1e-6)
            _p2_total    = _p2_feed_CO2 + _p2_feed_CO + _p2_feed_H2

            # Feed node
            self.mem2.nf_r_CO2[1].set_value(_p2_feed_CO2)
            self.mem2.nf_r_CO[1].set_value(_p2_feed_CO)
            self.mem2.nf_r_H2[1].set_value(_p2_feed_H2)

            # For Polyimide (H2-selective): most H2 permeates, CO2/CO stay in retentate
            _ret2_H2  = max(_p2_feed_H2  * 0.05, 1e-6)   # ~5% H2 retained
            _ret2_CO2 = max(_p2_feed_CO2 * 0.80, 1e-6)
            _ret2_CO  = max(_p2_feed_CO  * 0.80, 1e-6)
            _perm2_H2 = max(_p2_feed_H2  * 0.95, 1e-6)

            # Seed retentate nodes with linear interpolation
            for _k in range(1, _N2 + 2):
                _frac = (_k - 1) / _N2
                self.mem2.nf_r_CO2[_k].set_value(
                    _p2_feed_CO2 + _frac * (_ret2_CO2 - _p2_feed_CO2)
                )
                self.mem2.nf_r_CO[_k].set_value(
                    _p2_feed_CO + _frac * (_ret2_CO - _p2_feed_CO)
                )
                self.mem2.nf_r_H2[_k].set_value(
                    _p2_feed_H2 + _frac * (_ret2_H2 - _p2_feed_H2)
                )

            # Mole fractions (feed end as initial guess)
            _y0_CO2 = max(_p2_feed_CO2 / max(_p2_total, 1e-9), 1e-8)
            _y0_CO  = max(_p2_feed_CO  / max(_p2_total, 1e-9), 1e-8)
            _y0_H2  = max(_p2_feed_H2  / max(_p2_total, 1e-9), 1e-8)
            for _k in range(1, _N2 + 1):
                self.mem2.y_r_CO2[_k].set_value(_y0_CO2)
                self.mem2.y_r_CO[_k].set_value(_y0_CO)
                self.mem2.y_r_H2[_k].set_value(_y0_H2)
                # Permeate: H2-enriched
                _yp_H2 = min(max(_y0_H2 * 3.0, 1e-8), 0.98)
                self.mem2.y_p_H2[_k].set_value(_yp_H2)
                self.mem2.y_p_CO2[_k].set_value(max((1.0 - _yp_H2) * 0.8, 1e-8))
                self.mem2.y_p_CO[_k].set_value(max((1.0 - _yp_H2) * 0.2, 1e-8))

            # Permeate totals
            self.mem2.nf_p_H2_out.set_value(_perm2_H2)
            self.mem2.nf_p_CO2_out.set_value(max(_p2_feed_CO2 * 0.20, 1e-6))
            self.mem2.nf_p_CO_out.set_value(max(_p2_feed_CO * 0.20, 1e-6))
            self.mem2.stage_cut.set_value(max(min(_perm2_H2 / max(_p2_total, 1e-9), 0.99), 0.01))

            # Retentate pressure: start at 5 bar
            _p_r2_init = 5e5  # Pa
            self.mem2.p_r.set_value(_p_r2_init)

            # Mem2 area (rough estimate)
            _A_mem2 = max(_A_mem_val * 0.1, 100.0)
            self.mem2.A_mem.set_value(_A_mem2)

            # Mem2 cost
            _mem2_CAPEX = _BC_mem * _A_mem2
            _mem2_OPEX  = _mem2_CAPEX / (_t_life_mem * _op_hours) + _maint_mem * _mem2_CAPEX / _op_hours
            self.mem2.CAPEX.set_value(max(_mem2_CAPEX, 1.0))
            self.mem2.OPEX.set_value(max(_mem2_OPEX, 1e-3))

            # Flux seed (zero-initialised variables in GasPermeationMembrane)
            _J2_avg = _perm2_H2 / _N2
            for _k in range(1, _N2 + 1):
                self.mem2.J_CO2[_k].set_value(max(_p2_feed_CO2 * 0.20 / _N2, 1e-9))
                self.mem2.J_CO[_k].set_value(max(_p2_feed_CO * 0.20 / _N2, 1e-9))
                self.mem2.J_H2[_k].set_value(max(_J2_avg, 1e-9))

            # Second compressor seed
            # Read gamma and eta from the first compressor stage (same settings)
            _gamma2   = pyo.value(self.comp._comps[0].gamma)
            _eta_c2   = pyo.value(self.comp._comps[0].eta)
            _Tin2     = pyo.value(self.comp._comps[0].Tin)  # intercooler outlet T
            self.comp2_nf_total.set_value(_p2_total)
            _p_r2_bar = _p_r2_init / 1e5
            _Tout2 = _Tin2 + (_Tin2 / _eta_c2) * (_p_r2_bar ** ((_gamma2 - 1.0) / _gamma2) - 1.0)
            self.comp2_Tout.set_value(max(_Tout2, _Tin2 + 1.0))
            _pw2 = max(_p2_total * 35.0 * (_Tout2 - _Tin2) / 1000.0, 0.1)
            self.comp2_power_kw.set_value(_pw2)
            _bc2 = 23000.0 * (_pw2 / 73.55) ** 0.77
            self.comp2_base_cost.set_value(max(_bc2, 1.0))
            _fa2 = 776.5 * (_bc2 + 1e-6) ** 0.493
            self.comp2_Fa_cost.set_value(max(_fa2, 1.0))
            _inst2 = 1.565 * (_bc2 + _fa2)
            self.comp2_installed.set_value(max(_inst2, 1.0))
            _capex2 = _Econ_inst.convert_usd_to_euro * 1.38 * _inst2
            self.comp2_capex.set_value(max(_capex2, 1.0))
            _opex2 = _pw2 * _op_hours * _Econ_inst.price_power + _Econ_inst.maintenance_factor_rotating * _capex2
            self.comp2_opex.set_value(max(_opex2, 1.0))

        # --- Seed compression sub-model variables ---
        _with_flash_c = hasattr(self, "psi_f")
        _Pout_bar = pyo.value(self.comp.Pout_var)
        # (mem.p_r was already seeded early in the membrane section above)

        # 2. Compressor stages: Tout → power_kw → Guthrie cost chain
        _n_stg = len(self.comp._comps)
        for _k, _ck in enumerate(self.comp._comps):
            _Pin_k  = pyo.value(_ck.Pin) if _k == 0 else pyo.value(self.comp._P_inter[_k - 1])
            _Pout_k = pyo.value(self.comp._P_inter[_k]) if _k < _n_stg - 1 else _Pout_bar
            _Tin_k  = pyo.value(_ck.Tin) if _k == 0 else pyo.value(self.comp._T_ic_out[_k - 1])
            _g   = pyo.value(_ck.gamma)
            _eta = pyo.value(_ck.eta)
            _Tout_k = _Tin_k * (1.0 + (1.0 / _eta) * ((_Pout_k / max(_Pin_k, 1e-9)) ** ((_g - 1) / _g) - 1.0))
            _ck.Tout.set_value(_Tout_k)
            # Seed T_hot for the associated MixtureCp model so delta_h_mix evaluates correctly
            if _k == _n_stg - 1:
                self.comp.cp_hx.T_hot.set_value(_Tout_k)
            else:
                self.comp._cp_ic[_k].T_hot.set_value(_Tout_k)
            # power_kw — use n_total_elec_out (the "act" constraint in CombinedFlowsheet
            # replaces the fixed molar_flow param with the actual electrolyzer outlet flow)
            _cp_mix = (pyo.value(self.comp.cp_hx.cp_mix) if _k == _n_stg - 1
                       else pyo.value(self.comp._cp_ic[_k].cp_mix))
            _pow_k = pyo.value(self.n_total_elec_out) * _cp_mix * (_Tout_k - _Tin_k) / 1000.0
            _ck.power_kw.set_value(max(_pow_k, 1.0))
            # Guthrie cost chain: base_cost → F_a → installed → capex_total
            _base = 23000.0 * (max(_pow_k, 1.0) / 73.55) ** 0.77
            _fa   = 776.5 * _base ** 0.493
            _inst = pyo.value(_ck.f_install) * (_base + _fa)
            _cap  = pyo.value(_ck.usd_to_eur) * pyo.value(_ck.f_indirect) * _inst
            _ck.base_cost_comp.set_value(max(_base, 1.0))
            _ck.F_a_cost_comp.set_value(max(_fa, 1.0))
            _ck.installed_cost_comp.set_value(max(_inst, 1.0))
            _ck.capex_total.set_value(max(_cap, 1.0))

        # 3. Intercoolers: duty → area → Guthrie HX capex
        for _k, (_ic_k, _cp_k) in enumerate(zip(self.comp._ics, self.comp._cp_ic)):
            _dh_ic   = pyo.value(_cp_k.delta_h_mix)
            _duty_ic = max(pyo.value(self.n_total_elec_out) * _dh_ic, 0.1)
            _ic_k.duty_kw.set_value(_duty_ic)
            _area_ic = max(_duty_ic * 1000.0 / pyo.value(_ic_k.UA), 0.1)
            _ic_k.area_m2.set_value(_area_ic)
            _cp_p = 101.3 * (_area_ic * 10.764) ** 0.65
            _ic_k.capex_purchased.set_value(max(_cp_p, 1.0))
            _cap_ic = (_cp_p * pyo.value(_ic_k.cepci_ratio) * 3.29
                       * pyo.value(_ic_k.f_indirect) * pyo.value(_ic_k.usd_to_eur))
            _ic_k.capex_total.set_value(max(_cap_ic, 1.0))

        # 4. HX1 (final water cooler): duty → area → Guthrie capex
        _dh_hx  = pyo.value(self.comp.cp_hx.delta_h_mix)
        _duty_hx = max(pyo.value(self.n_total_elec_out) * _dh_hx, 0.1)
        self.comp.hx1.duty_kw.set_value(_duty_hx)
        _area_hx = max(_duty_hx * 1000.0 / pyo.value(self.comp.hx1.UA), 0.1)
        self.comp.hx1.area_m2.set_value(_area_hx)
        _cp_p_hx = 101.3 * (_area_hx * 10.764) ** 0.65
        self.comp.hx1.capex_purchased.set_value(max(_cp_p_hx, 1.0))
        _cap_hx = (_cp_p_hx * pyo.value(self.comp.hx1.cepci_ratio) * 3.29
                   * pyo.value(self.comp.hx1.f_indirect) * pyo.value(self.comp.hx1.usd_to_eur))
        self.comp.hx1.capex_total.set_value(max(_cap_hx, 1.0))

        # Compression component opex_annual  (avoids None → 0 in IPOPT)
        for _comp_k in self.comp._comps:
            _pw = pyo.value(_comp_k.power_kw)
            _ct = pyo.value(_comp_k.capex_total)
            _oa = (_pw * pyo.value(_comp_k.op_hours) * pyo.value(_comp_k.elec_price)
                   + pyo.value(_comp_k.maintenance_factor) * _ct)
            _comp_k.opex_annual.set_value(max(_oa, 0.0))

        for _ic_k in self.comp._ics:
            _dt = pyo.value(_ic_k.duty_kw)
            _ct = pyo.value(_ic_k.capex_total)
            _oa = (_dt * pyo.value(_ic_k.op_hours) * pyo.value(_ic_k.cooling_cost_kwh)
                   + pyo.value(_ic_k.maintenance_factor) * _ct)
            _ic_k.opex_annual.set_value(max(_oa, 0.0))

        _hx_list = [self.comp.hx1]
        if hasattr(self, "hx2"):
            _hx_list.append(self.hx2)
        for _hx in _hx_list:
            _dt = pyo.value(_hx.duty_kw)
            _ct = pyo.value(_hx.capex_total)
            _oa = (_dt * pyo.value(_hx.op_hours) * pyo.value(_hx.cooling_cost_kwh)
                   + pyo.value(_hx.maintenance_factor) * _ct)
            _hx.opex_annual.set_value(max(_oa, 0.0))

        # Compression aggregated cost
        _comp_tcap  = (
            sum(pyo.value(ck.capex_total)    for ck in self.comp._comps)
            + sum(pyo.value(ic.capex_total)  for ic in self.comp._ics)
            + pyo.value(self.comp.hx1.capex_total)
            + (pyo.value(self.hx2.capex_total) if hasattr(self, "hx2") else 0.0)
        )
        _comp_topex = (
            sum(pyo.value(ck.opex_annual)    for ck in self.comp._comps)
            + sum(pyo.value(ic.opex_annual)  for ic in self.comp._ics)
            + pyo.value(self.comp.hx1.opex_annual)
            + (pyo.value(self.hx2.opex_annual) if hasattr(self, "hx2") else 0.0)
        )
        self.comp.total_capex.set_value(max(_comp_tcap, 0.0))
        self.comp.total_opex.set_value(max(_comp_topex, 0.0))
        _comp_tlc = _comp_tcap + _plant_life * _comp_topex
        self.comp.total_lifecycle_cost.set_value(max(_comp_tlc, 0.0))

        # Syngas feedstock costs
        _sec_per_yr = _op_hours * 3600.0
        if hasattr(self, "_syngas_target_set"):
            _price_CO2  = pyo.value(self.price_CO2_feed_param)
            _cost_CO2   = _price_CO2 * pyo.value(self.n_CO2_fresh) * _sec_per_yr
            self.cost_CO2_feed.set_value(max(_cost_CO2, 0.0))
            if hasattr(self, "nf_H2_buy"):
                _price_H2  = pyo.value(self.price_H2_buy_param)
                _cost_H2   = _price_H2 * pyo.value(self.nf_H2_buy) * _sec_per_yr
                self.cost_H2_buy.set_value(max(_cost_H2, 0.0))
            # Syngas mass flow — CO + H2 only
            _mf_sg = (
                pyo.value(self.nf_CO_syngas) * _Const.MW_CO
                + pyo.value(self.nf_H2_syngas) * _Const.MW_H2
            )
            self.mf_syngas.set_value(max(_mf_sg, 1e-9))

        # CombinedFlowsheet aggregated cost
        _with_flash = hasattr(self, "psi_f")
        _fs_cap = pyo.value(self.elec.CAPEX) + _comp_tcap + pyo.value(self.mem.CAPEX)
        if _with_flash:
            _fs_cap += pyo.value(self.flash_CAPEX)
        if self._recycle_permeate_H2:
            _fs_cap += pyo.value(self.mem2.CAPEX) + pyo.value(self.comp2_capex)
        if self._dual_membrane:
            _fs_cap += pyo.value(self.mem_s2.CAPEX)

        _feedstock_opex_yr = 0.0
        if hasattr(self, "_syngas_target_set"):
            _feedstock_opex_yr = pyo.value(self.cost_CO2_feed)
            if hasattr(self, "nf_H2_buy"):
                _feedstock_opex_yr += pyo.value(self.cost_H2_buy)

        _fs_opex = (
            pyo.value(self.elec.OPEX) * _op_hours
            + _comp_topex
            + pyo.value(self.mem.OPEX) * _op_hours
            + _feedstock_opex_yr
        )
        if _with_flash:
            _fs_opex += pyo.value(self.flash_opex_annual)
        if self._recycle_permeate_H2:
            _fs_opex += pyo.value(self.mem2.OPEX) * _op_hours + pyo.value(self.comp2_opex)
        if self._dual_membrane:
            _fs_opex += pyo.value(self.mem_s2.OPEX) * _op_hours

        self.total_capex.set_value(max(_fs_cap, 0.0))
        self.total_opex.set_value(max(_fs_opex, 0.0))
        self.total_lifecycle_cost.set_value(max(_fs_cap + _plant_life * _fs_opex, 0.0))

    def solve(
        self,
        solver_name: str = "ipopt",
        show_log: bool = False,
        solver_options: dict = None,
        multistart: bool = False,
        multistart_kwargs: dict = None,
    ):
        """Solve the combined NLP.

        Parameters
        ----------
        solver_name : str
            Solver executable name (default "ipopt").
        show_log : bool
            Stream solver output to stdout.
        multistart : bool
            If True, solve via ``pyomo.contrib.multistart`` (repeated IPOPT
            restarts, best objective kept) for a pseudo-global solution.
            Run a normal solve first to seed a good incumbent.
        multistart_kwargs : dict, optional
            Overrides for the multistart wrapper (e.g. iterations, strategy).
        solver_options : dict, optional
            Override / extend any IPOPT option.  The defaults below are chosen
            to handle the large scale differences in this model:

            nlp_scaling_method = "gradient-based"
                IPOPT computes scaling factors from first-order derivatives.
                Critical when model variables span many orders of magnitude
                (Pa pressures, mol/s flows, € costs).
            max_iter = 5000
                Allow more iterations for the large coupled system.
            tol = 1e-6 / acceptable_tol = 1e-4 / acceptable_iter = 10
                Accept a solution once 10 consecutive iterations all satisfy
                the looser tolerance — avoids crashing on last digits.
            mu_strategy = "adaptive"
                Adaptive barrier parameter update (more robust than monotone).
            bound_push / bound_frac = 1e-8
                Start variables very close to their bounds (helps when many
                variables are near bounds at the optimum).
        """
        # --- Add Pyomo variable scaling factors (exported to IPOPT as nlp_scaling_method=user) ---
        # Lifecycle cost is O(1e7 €), physical flows are O(1–100 mol/s), pressures O(1e5–1e6 Pa).
        # Without explicit scaling the dual variables for cost constraints are O(1e-7) while
        # those for flow constraints are O(1), causing inf_du ~ 1e7 that prevents convergence.
        if not hasattr(self, "scaling_factor"):
            self.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        sf = self.scaling_factor
        # Cost variables — scale to O(1) in M€ or k€
        sf[self.total_lifecycle_cost] = 1e-6   # ~1e7 € → ~10
        sf[self.total_capex]          = 1e-5   # ~1e6 € → ~10
        sf[self.total_opex]           = 1e-4   # ~1e5 €/yr → ~10
        if hasattr(self, "cost_CO2_feed"):
            sf[self.cost_CO2_feed]    = 1e-4
        if hasattr(self, "cost_H2_buy"):
            sf[self.cost_H2_buy]      = 1e-4
        sf[self.elec.CAPEX]           = 1e-5
        sf[self.elec.OPEX]            = 1e-1   # €/h ~O(10)
        sf[self.elec.Power]           = 1e-4   # W  ~O(1e4-1e6)
        sf[self.elec.I]               = 1e-5   # A  ~O(1e5)
        sf[self.elec.n]               = 1e-1   # millions of cells ~O(10-100)
        sf[self.elec.total_area]      = 1e-4   # m² ~O(1e4)
        sf[self.mem.p_r]              = 1e-5   # Pa ~O(1e5-3e6)
        sf[self.mem.A_mem]            = 1e-2   # m² ~O(100-1000)
        sf[self.mem.CAPEX]            = 1e-3   # € ~O(1e3)
        sf[self.comp.total_capex]     = 1e-5
        sf[self.comp.total_opex]      = 1e-4
        for _ck in self.comp._comps:
            sf[_ck.capex_total]       = 1e-5
            sf[_ck.base_cost_comp]    = 1e-4
            sf[_ck.F_a_cost_comp]     = 1e-4
            sf[_ck.installed_cost_comp] = 1e-4
            sf[_ck.power_kw]          = 1e-2   # kW ~O(10-1000)
        for _ic in self.comp._ics:
            sf[_ic.capex_total]       = 1e-5
            sf[_ic.capex_purchased]   = 1e-4
            sf[_ic.duty_kw]           = 1e-2
            sf[_ic.area_m2]           = 1e-2
        sf[self.comp.hx1.capex_total]     = 1e-5
        sf[self.comp.hx1.capex_purchased] = 1e-4
        sf[self.comp.hx1.duty_kw]         = 1e-2
        sf[self.comp.hx1.area_m2]         = 1e-2
        if self._recycle_permeate_H2:
            sf[self.mem2.p_r]          = 1e-5
            sf[self.mem2.A_mem]        = 1e-2
            sf[self.mem2.CAPEX]        = 1e-3
            sf[self.comp2_power_kw]    = 1e-2
            sf[self.comp2_capex]       = 1e-5
            sf[self.comp2_base_cost]   = 1e-4
            sf[self.comp2_Fa_cost]     = 1e-4
            sf[self.comp2_installed]   = 1e-4
        if self._dual_membrane:
            sf[self.mem_s2.p_r]        = 1e-5
            sf[self.mem_s2.A_mem]      = 1e-2
            sf[self.mem_s2.CAPEX]      = 1e-3
        if hasattr(self, "h2_elec"):
            sf[self.h2_elec.CAPEX]                      = 1e-5   # ~1e6 €
            sf[self.h2_elec.OPEX]                       = 1e-4   # ~1e5 €/yr
            sf[self.h2_elec.MAINTENANCE]                = 1e-4
            sf[self.h2_elec.power_hp_mw]               = 1e-1   # ~10-1000 MW
            sf[self.h2_elec.x_hp]                       = 1e0    # O(2-7)
            sf[self.h2_elec.capex_electrolyzer_per_kw] = 1e-3   # ~O(1000 €/kW)
            sf[self.h2_elec.H2_out_kg_per_sec]         = 1e1    # ~0.01-1 kg/s
            sf[self.h2_elec.H2_out_mol_per_sec]        = 1e-1   # ~1-100 mol/s

        _defaults = {
            "nlp_scaling_method": "gradient-based",
            "max_iter": 5000,
            "tol": 1e-5,
            "acceptable_tol": 1.0,
            "acceptable_dual_inf_tol": 1e6,
            "acceptable_constr_viol_tol": 1.0,
            "acceptable_compl_inf_tol": 1e6,
            "acceptable_iter": 5,
            "mu_strategy": "adaptive",
            "mu_init": 1e-1,
            "bound_push": 1e-4,
            "bound_frac": 1e-4,
            "print_level": 5 if show_log else 0,
        }
        if solver_options:
            _defaults.update(solver_options)

        return run_solver(
            self, solver_name, _defaults, show_log=show_log,
            multistart=multistart, multistart_kwargs=multistart_kwargs,
        )

    def report(self):
        """Print a structured summary of the combined flowsheet solution."""
        with_flash = hasattr(self, "psi_f")
        N_mem      = self.mem._N
        _econ      = Economics()
        op_hours   = _econ.operating_hours_per_year
        plant_life = _econ.plant_life_years

        sep = "=" * 68
        print("\n" + sep)
        print(
            f"  MEMBRANE FLOWSHEET SUMMARY"
            f"  [Electrolyzer -> Compression"
            f"{' -> Flash' if with_flash else ''} -> Membrane]"
        )
        print(sep)

        # -------------------------------------------------------------------
        print("\n[ELECTROLYZER]")
        print(f"  E_appl          : {pyo.value(self.elec.E_appl):.4f}  V")
        print(
            f"  CD              : {pyo.value(self.elec.CD):.2f}  A/m²"
            f"  ({pyo.value(self.elec.CD) * 0.1:.2f} mA/cm²)"
        )
        print(f"  v               : {pyo.value(self.elec.v):.4f}  m/s")
        print(f"  n (cells)       : {pyo.value(self.elec.n) * 1e6:.3e}")
        print(f"  Power           : {pyo.value(self.elec.Power) / 1e6:.3f}  MW")
        print(f"  CO2share_in     : {pyo.value(self.elec.CO2share):.4f}")
        print(
            f"  n_CO2_fresh     : {pyo.value(self.n_CO2_fresh):.3f}  mol/s"
        )
        print(
            f"  nf_CO2_in       : {pyo.value(self.elec.nf_CO2_elec_in):.3f}  mol/s"
        )
        print(
            f"  nf_CO_in        : {pyo.value(self.elec.nf_CO_elec_in):.3f}  mol/s"
        )
        print(
            f"  nf_H2_in        : {pyo.value(self.elec.nf_H2_elec_in):.3f}  mol/s  (recycle pass-through)"
        )
        print(
            f"  nf_CO2_out      : {pyo.value(self.elec.nf_CO2_elec_out):.3f}  mol/s"
        )
        print(
            f"  nf_CO_out       : {pyo.value(self.elec.nf_CO_elec_out):.3f}  mol/s"
        )
        print(
            f"  nf_H2_out       : {pyo.value(self.elec.nf_H2_elec_out):.3f}  mol/s"
        )
        print(
            f"  Total cathode   : {pyo.value(self.n_total_elec_out):.3f}  mol/s"
        )
        print(f"  lamda_CO2       : {pyo.value(self.elec.lamda_CO2):.4f}")
        print(f"  Vcell           : {pyo.value(self.elec.Vcell):.4f}  V")
        print(f"  total_area      : {pyo.value(self.elec.total_area):,.1f}  m²")
        print(f"  CAPEX           : €{pyo.value(self.elec.CAPEX):,.0f}")
        print(
            f"  OPEX            : €{pyo.value(self.elec.OPEX) * op_hours:,.0f}/yr"
        )

        # -------------------------------------------------------------------
        print("\n[COMPRESSION FLOWSHEET]")
        self.comp.report()

        # -------------------------------------------------------------------
        if with_flash:
            print("\n[FLASH DRUM]")
            print(
                f"  Vapour fraction (ψ)  : {pyo.value(self.psi_f):.4f}"
            )
            print(
                f"  Feed composition     : "
                f"CO2 = {pyo.value(self.z_f_CO2):.4f}  "
                f"CO = {pyo.value(self.z_f_CO):.4f}  "
                f"H2 = {pyo.value(self.z_f_H2):.4f}"
            )
            print(
                f"  Liquid fractions x_i : "
                f"CO2 = {pyo.value(self.x_f_CO2):.4f}  "
                f"CO = {pyo.value(self.x_f_CO):.4f}  "
                f"H2 = {pyo.value(self.x_f_H2):.6f}"
            )
            print(
                f"  Vapour fractions y_i : "
                f"CO2 = {pyo.value(self.y_f_CO2):.4f}  "
                f"CO = {pyo.value(self.y_f_CO):.4f}  "
                f"H2 = {pyo.value(self.y_f_H2):.4f}"
            )
            print(
                f"  Vapour flows (mol/s) : "
                f"CO2 = {pyo.value(self.nf_fv_CO2):.3f}  "
                f"CO = {pyo.value(self.nf_fv_CO):.3f}  "
                f"H2 = {pyo.value(self.nf_fv_H2):.3f}"
            )
            print(
                f"  Liquid flows (mol/s) : "
                f"CO2 = {pyo.value(self.nf_fl_CO2):.3f}  "
                f"CO = {pyo.value(self.nf_fl_CO):.3f}  "
                f"H2 = {pyo.value(self.nf_fl_H2):.6f}"
            )
            print(
                f"  Vessel volume        : {pyo.value(self.flash_vol):.3f}  m³"
            )
            print(f"  CAPEX                : €{pyo.value(self.flash_CAPEX):,.0f}")
            print(
                f"  OPEX/yr              : €{pyo.value(self.flash_opex_annual):,.0f}"
            )

            print("\n[HX2 — NH3 REFRIGERATION]")
            print(f"  Inlet  T : {pyo.value(self.hx2.Tin) - 273.15:.2f} °C")
            print(f"  Outlet T : {pyo.value(self.hx2.Tout_target) - 273.15:.2f} °C")
            print(f"  ΔH_mix (sensible) : {pyo.value(self.comp.cp_hx.delta_h_mix2):.4f} kJ/mol")
            print(f"  CO2 condensed     : {pyo.value(self.nf_fl_CO2):.4f} mol/s"
                  f"  (ΔHvap = {pyo.value(self.delta_H_vap_CO2):.1f} kJ/mol)")
            print(f"  Duty (total)      : {pyo.value(self.hx2.duty_kw):.2f} kW_th")
            print(f"  CAPEX    : €{pyo.value(self.hx2.capex_total):,.0f}")
            print(f"  OPEX/yr  : €{pyo.value(self.hx2.opex_annual):,.0f}")

        # -------------------------------------------------------------------
        N_ret     = N_mem + 1
        co2_ret   = pyo.value(self.mem.nf_r_CO2[N_ret])
        co_ret    = pyo.value(self.mem.nf_r_CO[N_ret])
        h2_ret    = pyo.value(self.mem.nf_r_H2[N_ret])
        total_ret = co2_ret + co_ret + h2_ret
        purity    = (co_ret + h2_ret) / (total_ret + 1e-12) * 100.0

        print("\n[MEMBRANE]")
        print(f"  Material         : {self.mem.membrane_material}")
        print(f"  A_mem            : {pyo.value(self.mem.A_mem):.2f}  m²")
        print(f"  p_r              : {pyo.value(self.mem.p_r) / 1e5:.2f}  bar")
        print(f"  p_p              : {pyo.value(self.mem.p_p) / 1e5:.4f}  bar")
        print(f"  stage_cut        : {pyo.value(self.mem.stage_cut):.4f}")
        _pf_label = " (optimised)" if self._float_purge else " (fixed)"
        print(f"  purge_fraction   : {pyo.value(self.purge_fraction):.4f}{_pf_label}")
        print(f"  Retentate CO2    : {co2_ret:.3f}  mol/s")
        print(f"  Retentate CO     : {co_ret:.3f}  mol/s")
        print(f"  Retentate H2     : {h2_ret:.3f}  mol/s")
        _depletion_target = getattr(self.mem, "_co2_depletion_target", None)
        _target_str = f"{_depletion_target:.1f} %" if _depletion_target is not None else "none"
        print(f"  CO+H2 purity     : {purity:.3f} %  (CO2 depletion target: {_target_str})")
        print(
            f"  Permeate CO2     : {pyo.value(self.mem.nf_p_CO2_out):.3f}  mol/s  (-> recycle)"
        )
        print(
            f"  Permeate CO      : {pyo.value(self.mem.nf_p_CO_out):.3f}  mol/s  (-> recycle)"
        )
        if self._dual_membrane:
            _h2_dest = "-> syngas H2"
        elif self._recycle_permeate_H2:
            _h2_dest = "-> comp2/mem2"
        else:
            _h2_dest = "-> recycle (pass-through)"
        print(
            f"  Permeate H2      : {pyo.value(self.mem.nf_p_H2_out):.3f}  mol/s  ({_h2_dest})"
        )
        print(f"  CAPEX            : €{pyo.value(self.mem.CAPEX):,.0f}")
        print(f"  OPEX             : €{pyo.value(self.mem.OPEX) * op_hours:,.0f}/yr")
        if self._dual_membrane:
            _N_s2 = self.mem_s2._N
            _co2_s2_ret = pyo.value(self.mem_s2.nf_r_CO2[_N_s2 + 1])
            _co_s2_ret  = pyo.value(self.mem_s2.nf_r_CO[_N_s2 + 1])
            _h2_s2_ret  = pyo.value(self.mem_s2.nf_r_H2[_N_s2 + 1])
            print(f"\n[STAGE-2 MEMBRANE ({self.mem_s2.membrane_material} — CO2 separation)]")
            print(f"  A_mem            : {pyo.value(self.mem_s2.A_mem):.2f}  m²")
            print(f"  p_r              : {pyo.value(self.mem_s2.p_r)/1e5:.2f}  bar")
            print(f"  p_p              : {pyo.value(self.mem_s2.p_p)/1e5:.4f}  bar")
            print(f"  stage_cut        : {pyo.value(self.mem_s2.stage_cut):.4f}")
            print(f"  Feed CO2         : {pyo.value(self.mem_s2.nf_r_CO2[1]):.3f}  mol/s")
            print(f"  Feed CO          : {pyo.value(self.mem_s2.nf_r_CO[1]):.3f}  mol/s")
            print(f"  Feed H2          : {pyo.value(self.mem_s2.nf_r_H2[1]):.3f}  mol/s")
            print(f"  Retentate CO2    : {_co2_s2_ret:.3f}  mol/s  (-> syngas impurity)")
            print(f"  Retentate CO     : {_co_s2_ret:.3f}  mol/s  (-> syngas CO)")
            print(f"  Retentate H2     : {_h2_s2_ret:.3f}  mol/s  (-> syngas H2)")
            print(f"  Permeate CO2     : {pyo.value(self.mem_s2.nf_p_CO2_out):.3f}  mol/s  (-> recycle)")
            print(f"  Permeate CO      : {pyo.value(self.mem_s2.nf_p_CO_out):.3f}  mol/s  (-> recycle)")
            print(f"  Permeate H2      : {pyo.value(self.mem_s2.nf_p_H2_out):.3f}  mol/s  (-> recycle, small)")
            print(f"  CAPEX            : €{pyo.value(self.mem_s2.CAPEX):,.0f}")
            print(f"  OPEX             : €{pyo.value(self.mem_s2.OPEX) * op_hours:,.0f}/yr")
        if self._recycle_permeate_H2:
            _N2 = self.mem2._N
            _co2_r2 = pyo.value(self.mem2.nf_r_CO2[_N2 + 1])
            _co_r2  = pyo.value(self.mem2.nf_r_CO[_N2 + 1])
            _h2_r2  = pyo.value(self.mem2.nf_r_H2[_N2 + 1])
            print("\n[SECOND COMPRESSOR (permeate booster)]")
            print(f"  Feed flow        : {pyo.value(self.comp2_nf_total):.3f}  mol/s")
            print(f"  Inlet / outlet P : 1.00 bar -> {pyo.value(self.mem2.p_r)/1e5:.2f}  bar")
            print(f"  Outlet T         : {pyo.value(self.comp2_Tout) - 273.15:.2f}  °C")
            print(f"  Power            : {pyo.value(self.comp2_power_kw):.2f}  kW")
            print(f"  CAPEX            : €{pyo.value(self.comp2_capex):,.0f}")
            print(f"  OPEX             : €{pyo.value(self.comp2_opex):,.0f}/yr")
            print("\n[SECOND MEMBRANE (Polyimide — H2 recovery)]")
            print(f"  A_mem            : {pyo.value(self.mem2.A_mem):.2f}  m²")
            print(f"  p_r              : {pyo.value(self.mem2.p_r)/1e5:.2f}  bar")
            print(f"  p_p              : {pyo.value(self.mem2.p_p)/1e5:.4f}  bar")
            print(f"  stage_cut        : {pyo.value(self.mem2.stage_cut):.4f}")
            print(f"  Permeate H2      : {pyo.value(self.mem2.nf_p_H2_out):.3f}  mol/s  (-> syngas)")
            print(f"  Permeate CO2     : {pyo.value(self.mem2.nf_p_CO2_out):.3f}  mol/s  (discarded)")
            print(f"  Permeate CO      : {pyo.value(self.mem2.nf_p_CO_out):.3f}  mol/s  (discarded)")
            print(f"  Retentate CO2    : {_co2_r2:.3f}  mol/s  (-> recycle)")
            print(f"  Retentate CO     : {_co_r2:.3f}  mol/s  (-> recycle)")
            print(f"  Retentate H2     : {_h2_r2:.3f}  mol/s  (discarded)")
            print(f"  CAPEX            : €{pyo.value(self.mem2.CAPEX):,.0f}")
            print(f"  OPEX             : €{pyo.value(self.mem2.OPEX) * op_hours:,.0f}/yr")

        # -------------------------------------------------------------------
        if hasattr(self, "_syngas_target_set"):
            _mf_t_per_day = pyo.value(self.mf_syngas) * 3600.0 * 24.0 / 1000.0
            _nf_CO  = pyo.value(self.nf_CO_syngas)
            _nf_H2  = pyo.value(self.nf_H2_syngas)
            _nf_CO2 = pyo.value(self.nf_CO2_syngas)
            _nf_tot = pyo.value(self.nf_syngas)
            print("\n[SYNGAS PRODUCT]")
            print(f"  H2 type          : {self._H2_type}")
            print(f"  H2/CO ratio      : {pyo.value(self.syngas_ratio_param):.2f}")
            print(f"  nf_CO_syngas     : {_nf_CO:.3f}  mol/s")
            print(f"  nf_H2_syngas     : {_nf_H2:.3f}  mol/s")
            print(f"  nf_CO2_syngas    : {_nf_CO2:.4f}  mol/s")
            print(f"  Total mole flow  : {_nf_tot:.3f}  mol/s")
            _co2_tpd = _nf_CO2 * 44.01e-3 * 3600 * 24 / 1000
            print(f"  Mass flow        : {_mf_t_per_day:.3f}  t/day  (CO+H2 only)")
            print(f"  CO2 impurity     : {_co2_tpd:.3f}  t/day  (not counted in target)")
            print(f"  Target mass flow : {pyo.value(self.mf_syngas_target)*3600*24/1000:.3f}  t/day")
            if self._H2_type == "Buy":
                print(f"  nf_H2_buy        : {pyo.value(self.nf_H2_buy):.3f}  mol/s")
                print(f"  Annual H2 cost   : €{pyo.value(self.cost_H2_buy):,.0f}/yr")
            elif self._H2_type == "H2Electrolyzer":
                _h2_kg_s = pyo.value(self.h2_elec.H2_out_kg_per_sec) or 0.0
                print("\n[H2 WATER ELECTROLYZER]")
                print(f"  Power            : {pyo.value(self.h2_elec.power_hp_mw):.2f}  MW")
                print(f"  H2 produced      : {pyo.value(self.h2_elec.H2_out_mol_per_sec):.3f}  mol/s")
                print(f"  H2 produced      : {_h2_kg_s * 3600 * 24 / 1000:.3f}  t/day")
                print(f"  CAPEX/kW         : €{pyo.value(self.h2_elec.capex_electrolyzer_per_kw):.0f}/kW")
                print(f"  CAPEX            : €{pyo.value(self.h2_elec.CAPEX):,.0f}")
                print(f"  OPEX             : €{pyo.value(self.h2_elec.OPEX):,.0f}/yr")
                print(f"  Maintenance      : €{pyo.value(self.h2_elec.MAINTENANCE):,.0f}/yr")
                print(f"  Cost of H2       : €{pyo.value(self.h2_elec.cost_per_kg_h2):.3f}/kg")
            print(f"  Annual CO2 cost  : €{pyo.value(self.cost_CO2_feed):,.0f}/yr")

        # -------------------------------------------------------------------
        # Cost breakdown tables
        # -------------------------------------------------------------------
        from prettytable import PrettyTable
        _cv = pyo.value

        # --- Collect CAPEX rows ---
        capex_rows = [
            ("CO2 Electrolyzer",      _cv(self.elec.CAPEX)),
            ("Compression flowsheet", _cv(self.comp.total_capex)),
            ("Membrane 1",            _cv(self.mem.CAPEX)),
        ]
        if with_flash:
            capex_rows.append(("Flash drum",        _cv(self.flash_CAPEX)))
            capex_rows.append(("HX2 (NH3 refrig.)", _cv(self.hx2.capex_total)))
        if self._recycle_permeate_H2:
            capex_rows.append(("Compressor 2",       _cv(self.comp2_capex)))
            capex_rows.append(("Membrane 2",         _cv(self.mem2.CAPEX)))
        if self._dual_membrane:
            capex_rows.append(("Membrane stage 2",   _cv(self.mem_s2.CAPEX)))
        if hasattr(self, "_syngas_target_set") and self._H2_type == "H2Electrolyzer":
            capex_rows.append(("H2 Electrolyzer",    _cv(self.h2_elec.CAPEX)))

        # --- Collect OPEX rows ---
        opex_rows = [
            ("CO2 Electrolyzer electricity+maint.", _cv(self.elec.OPEX) * op_hours),
            ("Compression flowsheet", _cv(self.comp.total_opex)),
            ("Membrane 1",            _cv(self.mem.OPEX) * op_hours),
        ]
        if with_flash:
            opex_rows.append(("Flash drum",        _cv(self.flash_opex_annual)))
            opex_rows.append(("HX2 (NH3 refrig.)", _cv(self.hx2.opex_annual)))
        if self._recycle_permeate_H2:
            opex_rows.append(("Compressor 2",       _cv(self.comp2_opex)))
            opex_rows.append(("Membrane 2",         _cv(self.mem2.OPEX) * op_hours))
        if self._dual_membrane:
            opex_rows.append(("Membrane stage 2",   _cv(self.mem_s2.OPEX) * op_hours))
        if hasattr(self, "_syngas_target_set"):
            opex_rows.append(("CO2 feed cost",      _cv(self.cost_CO2_feed)))
            if self._H2_type == "Buy":
                opex_rows.append(("H2 purchase cost",     _cv(self.cost_H2_buy)))
            elif self._H2_type == "H2Electrolyzer":
                opex_rows.append(("H2 Electrolyzer", (_cv(self.h2_elec.OPEX) or 0.0) + (_cv(self.h2_elec.MAINTENANCE) or 0.0)))

        def _cost_table(title, rows, unit):
            total = sum(v for _, v in rows)
            t = PrettyTable()
            t.field_names = ["Component", f"Cost ({unit})", "Share (%)"]
            t.align["Component"] = "l"
            t.align[f"Cost ({unit})"] = "r"
            t.align["Share (%)"] = "r"
            for name, val in rows:
                share = val / total * 100.0 if total > 0 else 0.0
                t.add_row([name, f"{val:,.0f}", f"{share:.1f}"])
            t.add_divider()
            t.add_row(["TOTAL", f"{total:,.0f}", "100.0"])
            print(f"\n{title}")
            print(t)

        _cost_table("CAPEX BREAKDOWN", capex_rows, "€")
        _cost_table("OPEX BREAKDOWN  (€/yr)", opex_rows, "€/yr")

        # -------------------------------------------------------------------
        print("\n" + sep)
        print(f"{'COMBINED FLOWSHEET TOTALS':^68}")
        print(sep)
        _total_capex = _cv(self.total_capex)
        _total_opex  = _cv(self.total_opex)
        _total_lc    = _cv(self.total_lifecycle_cost)
        print(f"  Total CAPEX                    : €{_total_capex:,.0f}")
        print(f"  Total Annual OPEX              : €{_total_opex:,.0f}/yr")
        print(f"  NPV Cost ({plant_life} yr, {_econ.interest_rate*100:.1f}%)          : €{_total_lc:,.0f}  [CAPEX + PVF×OPEX, PVF={_econ.pvf:.2f}]")
        if hasattr(self, "_syngas_target_set"):
            _mf_t_per_day = _cv(self.mf_syngas) * 3600.0 * 24.0 / 1000.0
            if _mf_t_per_day > 0:
                _lc_per_t = _total_lc / (_mf_t_per_day * 365.0 * plant_life)
                print(f"  Lifecycle cost / syngas        : €{_lc_per_t:,.2f}/t")
        print(sep + "\n")
